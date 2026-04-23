#!/usr/bin/env python3
"""
ss_run_with_prior.py
====================
원논문(주월랑, 2022) 통계를 사전 정보로 활용한 Bayesian PCM-SEM 실행기.

논문 요약 통계:
  N=86, 21문항 Likert 5점 척도
  X = 쓰기인식  M = 4.015 (SD 미보고 → 추정 적용)
  M = 쓰기반응  M = 3.348
  Y = 수행태도  M = 3.511
  성별: 여학생 59명(68.6%), 남학생 27명
  Cronbach α = 0.854  /  r(성별, 쓰기인식) = .290**

실행하는 모형:
  1. sem_pcm_v2.stan        → 약한 사전 분포  (비교 기준선)
  2. sem_pcm_with_prior.stan → 강한 사전 분포  (원논문 통계 반영)

생성 파일:
  ss_prior_data_N86.json          — Stan 입력 데이터 (재현용)
  ss_mcmc_weakprior_N86.csv       — 약한 사전 분포 MCMC 샘플
  ss_mcmc_strongprior_N86.csv     — 강한 사전 분포 MCMC 샘플
  ss_prior_comparison_N86.csv     — 두 모형 결과 비교표
  ss_prior_predictive_check.csv   — 사전 예측 점검 샘플 (prior_only=1)

요구사항:
  cmdstanpy >= 1.1.0  (pip install cmdstanpy --upgrade)
  CmdStan >= 2.33     (python -c "import cmdstanpy; cmdstanpy.install_cmdstan()")

실행:
  python ss_run_with_prior.py           # 기본 실행
  python ss_run_with_prior.py --ppc     # 사전 예측 점검만
  python ss_run_with_prior.py --weak    # 약한 사전 분포 모형만
  python ss_run_with_prior.py --chains 2 --sampling 500   # 빠른 테스트
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import cmdstanpy
    from cmdstanpy import CmdStanModel
    print(f"cmdstanpy {cmdstanpy.__version__} 로드됨")
except ImportError:
    print("ERROR: cmdstanpy 가 설치되지 않았습니다.")
    print("  pip install cmdstanpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))
STAN_WEAK      = os.path.join(OUT_DIR, 'sem_pcm_v2.stan')
STAN_STRONG    = os.path.join(OUT_DIR, 'sem_pcm_with_prior.stan')

# ─────────────────────────────────────────────────────────────────
# 원논문 통계 (주월랑, 2022)
# ─────────────────────────────────────────────────────────────────
PAPER_STATS = dict(
    N         = 86,
    n_female  = 59,          # 68.6%
    # 구인별 복합 점수 평균 (1-5 척도)
    mean_X    = 4.015,       # 쓰기인식
    mean_M    = 3.348,       # 쓰기반응
    mean_Y    = 3.511,       # 수행태도
    # 표준편차: 논문 미보고 → Likert 5점 척도 전형적 범위에서 추정
    sd_X      = 0.65,
    sd_M      = 0.72,
    sd_Y      = 0.68,
    # 보고된 상관
    r_gender_X = 0.290,      # 성별↔쓰기인식 (Pearson r, p<.01)
    # Cronbach α
    alpha_cronbach = 0.854,
)

# 문항 구성 (현재 시뮬레이션 설정)
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

# MCMC 기본 설정
SEED          = 2024
CHAINS        = 4
ITER_WARMUP   = 1000
ITER_SAMPLING = 1000
ADAPT_DELTA   = 0.92
MAX_TREEDEPTH = 12

# ─────────────────────────────────────────────────────────────────
# PCM 기댓값 계산 및 임계값 역산 (사전 분포 보정)
# ─────────────────────────────────────────────────────────────────
def pcm_expected_score(theta_val, c_offset, base=None, K=5):
    """
    주어진 θ값과 임계값 오프셋 c에서 PCM 기댓값 계산.
    thresholds = c_offset + base
    base는 대칭 분포: [-1.5, -0.5, 0.5, 1.5]
    """
    if base is None:
        base = np.array([-1.5, -0.5, 0.5, 1.5])
    deltas = c_offset + base
    lp = np.zeros(K)
    for k in range(1, K):
        lp[k] = lp[k-1] + (theta_val - deltas[k-1])
    lp -= lp.max()
    probs = np.exp(lp)
    probs /= probs.sum()
    return np.dot(probs, np.arange(1, K+1))

def find_threshold_offset(target_mean, theta_val=0.0, K=5, tol=1e-6):
    """
    이분 탐색: E[y | θ=theta_val] = target_mean 이 되는 c_offset 탐색.
    target_mean이 높을수록 c_offset은 낮음 (쉬운 문항 = 낮은 임계값).
    """
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = pcm_expected_score(theta_val, mid, K=K)
        if val > target_mean:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0

# ─────────────────────────────────────────────────────────────────
# 사전 분포 파라미터 보정
# ─────────────────────────────────────────────────────────────────
def calibrate_priors(stats=PAPER_STATS, verbose=True):
    """
    원논문 요약 통계에서 Stan informative prior 파라미터를 계산.

    핵심 논리:
    1. 복합 점수 평균 → PCM 임계값 오프셋 역산
       E[y_i | θ=0] = paper mean  →  c_X, c_M, c_Y
    2. 성별 효과:
       r(gender, X) = .290  →  gamma_M 사전 설정
    3. 경로 계수:
       직접 보고 없음 → 쓰기 태도 선행 연구 전형 값 사용
    4. 절편:
       α_M ≈ 0  (잔재 분산 = 1 하에서 구조 방정식 기준점)
    """
    p = stats
    female_ratio = p['n_female'] / p['N']   # 0.686

    # ── 1. 임계값 오프셋 역산 ────────────────────────────────
    c_X = find_threshold_offset(p['mean_X'])
    c_M = find_threshold_offset(p['mean_M'])
    c_Y = find_threshold_offset(p['mean_Y'])
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    delta_X = c_X + base   # X 구인 기준 임계값 벡터 (K-1=4개)
    delta_M = c_M + base   # M 구인 기준 임계값 벡터
    delta_Y = c_Y + base   # Y 구인 기준 임계값 벡터

    if verbose:
        print("=== 임계값 보정 ===")
        for name, c, m_target in [('X(쓰기인식)', c_X, p['mean_X']),
                                   ('M(쓰기반응)', c_M, p['mean_M']),
                                   ('Y(수행태도)', c_Y, p['mean_Y'])]:
            e = pcm_expected_score(0.0, c)
            print(f"  {name}: 목표평균={m_target:.3f}, c={c:.4f}, E[y|θ=0]={e:.4f}")

    # ── 2. 성별 효과 사전 분포 ───────────────────────────────
    # r(gender, X) = .290 이지만 이는 원논문의 관측 상관.
    # 우리 모형에서 gender는 M과 Y의 공변량이므로,
    # M에 대한 gender 효과를 유사한 크기(소~중간)로 설정.
    # r = 0.20~0.29 범위 → 표준화 계수 ≈ 0.15~0.25
    prior_gM_mu = 0.20    # 성별→M: 보통 수준 사전 평균
    prior_gM_sd = 0.20    # 불확실성 반영 (±0.40 범위 포함)
    prior_gY_mu = 0.10    # 성별→Y: 직접 근거 없어 약한 사전
    prior_gY_sd = 0.20

    # ── 3. 경로 계수 사전 분포 ───────────────────────────────
    # 쓰기 태도 하위 척도 간 관계:
    #   β₁ (쓰기인식→쓰기반응): 긍정적 인식이 반응(흥미·자신감)을 예측
    #        쓰기 태도 연구 전형 r ≈ 0.30~0.55 → β₁ 사전 평균 0.35
    #   β₂ (쓰기반응→수행태도): 흥미·자신감이 수행 태도를 예측
    #        β₂ 사전 평균 0.35
    #   γ₁ (쓰기인식→수행태도, 직접): 매개 후 잔여 직접 효과
    #        약한 사전 (완전 매개 가능성 포함)
    prior_b1_mu = 0.35;   prior_b1_sd = 0.20
    prior_b2_mu = 0.35;   prior_b2_sd = 0.20
    prior_g1_mu = 0.15;   prior_g1_sd = 0.20

    # ── 4. 절편 사전 분포 ────────────────────────────────────
    # PCM-SEM에서 잠재 변수의 실제 평균은 임계값으로 흡수됨.
    # α_M, α_Y는 구조 방정식 절편 → 약한 사전 (gender 효과로 편이 소)
    prior_aM_mu = 0.0;  prior_aM_sd = 1.0
    prior_aY_mu = 0.0;  prior_aY_sd = 1.0

    priors = dict(
        prior_b1_mu=prior_b1_mu, prior_b1_sd=prior_b1_sd,
        prior_b2_mu=prior_b2_mu, prior_b2_sd=prior_b2_sd,
        prior_g1_mu=prior_g1_mu, prior_g1_sd=prior_g1_sd,
        prior_aM_mu=prior_aM_mu, prior_aM_sd=prior_aM_sd,
        prior_aY_mu=prior_aY_mu, prior_aY_sd=prior_aY_sd,
        prior_gM_mu=prior_gM_mu, prior_gM_sd=prior_gM_sd,
        prior_gY_mu=prior_gY_mu, prior_gY_sd=prior_gY_sd,
        prior_delta_X=delta_X.tolist(),
        prior_delta_M=delta_M.tolist(),
        prior_delta_Y=delta_Y.tolist(),
        prior_delta_sd=0.30,    # 문항 간 임계값 변이 허용 범위
        prior_only=0,
    )

    if verbose:
        print("\n=== 경로 계수 사전 분포 ===")
        for name, mu, sd in [
            ('β₁ (X→M)', prior_b1_mu, prior_b1_sd),
            ('β₂ (M→Y)', prior_b2_mu, prior_b2_sd),
            ('γ₁ (X→Y 직접)', prior_g1_mu, prior_g1_sd),
            ('γ_M (성별→M)', prior_gM_mu, prior_gM_sd),
            ('γ_Y (성별→Y)', prior_gY_mu, prior_gY_sd),
        ]:
            print(f"  {name:18s}: N({mu:.2f}, {sd:.2f})")

    return priors

# ─────────────────────────────────────────────────────────────────
# PCM 데이터 생성 (원논문 통계 기반 캘리브레이션)
# ─────────────────────────────────────────────────────────────────
def pcm_sample_vectorized(theta, deltas, rng):
    N  = len(theta)
    lp = np.zeros((N, K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - deltas[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    gumbel = -np.log(-np.log(rng.uniform(size=(N, K)) + 1e-15))
    return np.argmax(lp + gumbel, axis=1) + 1

def generate_paper_like_data(priors, N=86, seed=SEED):
    """
    원논문 통계에서 보정된 임계값을 사용해 N=86 데이터 생성.
    구조 방정식은 약한 사전 중심값 (β₁=0.35, β₂=0.35, γ₁=0.15)으로 설정.
    """
    rng = np.random.default_rng(seed)

    female_ratio = PAPER_STATS['n_female'] / PAPER_STATS['N']
    gender = (rng.uniform(size=N) < female_ratio).astype(float)  # 1=여성

    # 구조 방정식 (사전 중심값 사용)
    b1, b2, g1    = priors['prior_b1_mu'], priors['prior_b2_mu'], priors['prior_g1_mu']
    gM, gY        = priors['prior_gM_mu'], priors['prior_gY_mu']
    aM, aY        = priors['prior_aM_mu'], priors['prior_aY_mu']

    tX = rng.standard_normal(N)
    tM = aM + b1*tX + gM*gender + rng.standard_normal(N)
    tY = aY + g1*tX + b2*tM + gY*gender + rng.standard_normal(N)

    # 문항 임계값: 구인별 보정값 + 문항 간 소잡음
    delta_X = np.array(priors['prior_delta_X'])
    delta_M = np.array(priors['prior_delta_M'])
    delta_Y = np.array(priors['prior_delta_Y'])
    d_sd    = priors['prior_delta_sd']

    item_deltas = []
    for _ in range(I_X): item_deltas.append(delta_X + rng.normal(0, d_sd, K-1))
    for _ in range(I_M): item_deltas.append(delta_M + rng.normal(0, d_sd, K-1))
    for _ in range(I_Y): item_deltas.append(delta_Y + rng.normal(0, d_sd, K-1))

    y = np.zeros((N, I), dtype=np.int32)
    for i in range(I):
        if   i < I_X:          theta_use = tX
        elif i < I_X + I_M:    theta_use = tM
        else:                  theta_use = tY
        y[:, i] = pcm_sample_vectorized(theta_use, item_deltas[i], rng)

    # 데이터 품질 확인
    composite = {
        'X': y[:, :I_X].mean(1).mean(),
        'M': y[:, I_X:I_X+I_M].mean(1).mean(),
        'Y': y[:, I_X+I_M:].mean(1).mean(),
    }
    print("\n=== 생성된 데이터 복합 점수 평균 ===")
    print(f"  X(쓰기인식): {composite['X']:.3f}  (논문: {PAPER_STATS['mean_X']:.3f})")
    print(f"  M(쓰기반응): {composite['M']:.3f}  (논문: {PAPER_STATS['mean_M']:.3f})")
    print(f"  Y(수행태도): {composite['Y']:.3f}  (논문: {PAPER_STATS['mean_Y']:.3f})")

    return y, tX, tM, tY, gender

# ─────────────────────────────────────────────────────────────────
# Stan 데이터 딕셔너리 구성
# ─────────────────────────────────────────────────────────────────
def make_stan_data_weak(y, gender):
    """약한 사전 분포 모형용 (sem_pcm_v2.stan)"""
    return {
        'N': y.shape[0], 'I': I, 'K': K, 'I_X': I_X, 'I_M': I_M,
        'y': y.tolist(), 'gender': gender.tolist(),
    }

def make_stan_data_strong(y, gender, priors, prior_only=0):
    """강한 사전 분포 모형용 (sem_pcm_with_prior.stan)"""
    d = {
        'N': y.shape[0], 'I': I, 'K': K, 'I_X': I_X, 'I_M': I_M,
        'y': y.tolist(), 'gender': gender.tolist(),
        'prior_only': prior_only,
    }
    d.update(priors)
    return d

# ─────────────────────────────────────────────────────────────────
# MCMC 실행
# ─────────────────────────────────────────────────────────────────
def run_mcmc(model, stan_data, label, chains=CHAINS,
             warmup=ITER_WARMUP, sampling=ITER_SAMPLING):
    print(f"\n{'─'*60}")
    print(f"MCMC 실행: {label}")
    print(f"  {chains}체인 × {warmup}워밍업 + {sampling}샘플링")
    t0 = time.time()

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=sampling,
        seed=SEED,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        show_progress=True,
        show_console=False,
    )
    elapsed = time.time() - t0
    print(f"완료: {elapsed:.1f}s ({elapsed/60:.1f}분)")
    return fit

# ─────────────────────────────────────────────────────────────────
# 결과 요약 및 비교
# ─────────────────────────────────────────────────────────────────
KEY_PARAMS = ['b1','b2','g1','gamma_M','gamma_Y','alpha_M','alpha_Y',
              'indirect_effect','total_effect','prop_mediated']

def summarize_fit(fit, label):
    """MCMC 사후 분포 요약 출력 + DataFrame 반환"""
    draws = fit.draws_pd()
    true_vals_approx = {
        'b1': 0.35, 'b2': 0.35, 'g1': 0.15,
        'gamma_M': 0.20, 'gamma_Y': 0.10,
        'alpha_M': 0.0, 'alpha_Y': 0.0,
        'indirect_effect': 0.35*0.35,
        'total_effect': 0.15 + 0.35*0.35,
        'prop_mediated': (0.35*0.35)/(0.15+0.35*0.35),
    }

    print(f"\n=== 사후 분포 요약: {label} ===")
    print(f"{'파라미터':22s} {'사후평균':>9s} {'사후SD':>8s} {'95%CI 하':>10s} {'95%CI 상':>10s} {'P(>0)':>7s}")
    print("─" * 70)

    rows = []
    for p in KEY_PARAMS:
        if p not in draws.columns:
            continue
        s = draws[p].values
        lo, hi = np.percentile(s, [2.5, 97.5])
        prob_pos = (s > 0).mean()
        print(f"  {p:20s} {s.mean():9.4f} {s.std():8.4f} {lo:10.4f} {hi:10.4f} {prob_pos:7.3f}")
        rows.append({
            'model': label, 'param': p,
            'mean': s.mean(), 'sd': s.std(),
            'ci_lo': lo, 'ci_hi': hi,
            'prob_pos': prob_pos,
        })

    df = pd.DataFrame(rows)
    return df, draws

def print_comparison(df_weak, df_strong):
    """두 모형 결과 나란히 비교"""
    print("\n" + "═"*80)
    print("모형 비교: 약한 사전 분포 (sem_pcm_v2) vs. 강한 사전 분포 (sem_pcm_with_prior)")
    print("═"*80)
    print(f"{'파라미터':22s} {'약한사전 평균':>13s} {'약한사전 95%CI':>18s}  "
          f"{'강한사전 평균':>13s} {'강한사전 95%CI':>18s}")
    print("─" * 90)
    for p in KEY_PARAMS:
        w = df_weak[df_weak.param == p]
        s = df_strong[df_strong.param == p]
        if len(w) == 0 or len(s) == 0:
            continue
        w = w.iloc[0]; s = s.iloc[0]
        print(f"  {p:20s}  {w['mean']:>10.4f}  [{w['ci_lo']:>7.4f},{w['ci_hi']:>7.4f}]   "
              f"{s['mean']:>10.4f}  [{s['ci_lo']:>7.4f},{s['ci_hi']:>7.4f}]")

def bayesian_mediation_report(draws_weak, draws_strong):
    """
    PCM-SEM의 핵심 기여: 베이지안 매개 효과 분석
    - P(β₁>0 ∧ β₂>0 | data): 인과 방향성 확률
    - 사후 분포에서 직접 계산한 간접 효과 분포
    - 매개 비율의 사후 분포
    """
    print("\n" + "═"*60)
    print("베이지안 매개 효과 분석")
    print("═"*60)

    for label, draws in [("약한 사전 분포", draws_weak),
                          ("강한 사전 분포", draws_strong)]:
        b1 = draws['b1'].values
        b2 = draws['b2'].values
        g1 = draws['g1'].values

        indirect = b1 * b2
        total    = g1 + indirect
        prop_med = np.where(np.abs(total) > 1e-10,
                            indirect / total, np.nan)

        p_dir = (b1 > 0) & (b2 > 0)   # 양방향 모두 양수
        ind_lo, ind_hi = np.percentile(indirect, [2.5, 97.5])
        prop_lo, prop_hi = np.nanpercentile(prop_med, [2.5, 97.5])

        print(f"\n  [{label}]")
        print(f"  P(β₁>0 AND β₂>0 | data)     = {p_dir.mean():.4f}  "
              f"← 인과 방향성 확률")
        print(f"  간접 효과 (β₁β₂) 사후평균    = {indirect.mean():.4f}  "
              f"95%CI: [{ind_lo:.4f}, {ind_hi:.4f}]")
        sig = "유의 (0 미포함)" if (ind_lo > 0 or ind_hi < 0) else "비유의 (0 포함)"
        print(f"  간접 효과 유의성              = {sig}")
        print(f"  P(간접효과 > 0 | data)        = {(indirect > 0).mean():.4f}")
        print(f"  매개 비율 사후평균            = {np.nanmean(prop_med):.4f}  "
              f"95%CI: [{prop_lo:.4f}, {prop_hi:.4f}]")
        print(f"  P(매개비율 > 0.5 | data)      = {(prop_med > 0.5).mean():.4f}  "
              f"← 완전 매개 가능성")

# ─────────────────────────────────────────────────────────────────
# 사전 예측 점검 (Prior Predictive Check)
# ─────────────────────────────────────────────────────────────────
def run_prior_predictive_check(model_strong, priors, N=86):
    """
    prior_only=1 로 실행: 데이터 없이 사전 분포에서만 샘플링.
    사전 분포가 합리적인 범위의 응답을 예측하는지 확인.
    """
    print("\n=== 사전 예측 점검 (Prior Predictive Check) ===")
    # 더미 응답 행렬 (실제로 사용되지 않음, prior_only=1)
    dummy_y = np.ones((N, I), dtype=int)
    dummy_gender = np.zeros(N)
    stan_data = make_stan_data_strong(dummy_y, dummy_gender, priors, prior_only=1)

    fit = model_strong.sample(
        data=stan_data,
        chains=2,
        iter_warmup=500,
        iter_sampling=500,
        seed=SEED,
        show_progress=False,
        show_console=False,
    )
    draws = fit.draws_pd()
    print("사전 분포 파라미터 범위:")
    for p in ['b1', 'b2', 'g1', 'indirect_effect']:
        if p in draws.columns:
            s = draws[p].values
            print(f"  {p:20s}: [{s.min():.3f}, {s.max():.3f}]  "
                  f"평균={s.mean():.3f}, SD={s.std():.3f}")

    ppc_file = os.path.join(OUT_DIR, 'ss_prior_predictive_check.csv')
    draws[[c for c in draws.columns
           if c in KEY_PARAMS + ['lp__']]].to_csv(ppc_file, index=False)
    print(f"저장: {ppc_file}")
    return draws

# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main(args):
    # 사전 분포 보정
    print("=" * 60)
    print("원논문 통계 기반 사전 분포 보정")
    print("=" * 60)
    priors = calibrate_priors(verbose=True)

    # 데이터 생성 (원논문 통계 기반)
    print("\n" + "=" * 60)
    print(f"데이터 생성: N={PAPER_STATS['N']}, 원논문 통계 기반")
    print("=" * 60)
    y, tX, tM, tY, gender = generate_paper_like_data(priors)

    # Stan 데이터 저장
    data_file = os.path.join(OUT_DIR, 'ss_prior_data_N86.json')
    save_data = make_stan_data_weak(y, gender)
    save_data['true_params_approx'] = {
        'b1': priors['prior_b1_mu'], 'b2': priors['prior_b2_mu'],
        'g1': priors['prior_g1_mu'], 'gamma_M': priors['prior_gM_mu'],
        'paper_mean_X': PAPER_STATS['mean_X'],
        'paper_mean_M': PAPER_STATS['mean_M'],
        'paper_mean_Y': PAPER_STATS['mean_Y'],
    }
    with open(data_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"데이터 저장: {data_file}")

    # ── 사전 예측 점검 ──────────────────────────────────────
    if args.ppc:
        model_strong = CmdStanModel(stan_file=STAN_STRONG)
        run_prior_predictive_check(model_strong, priors)
        if not (args.weak or args.strong):
            return

    all_results = []

    # ── 모형 1: 약한 사전 분포 (sem_pcm_v2.stan) ──────────
    if not args.strong:
        print(f"\n모형 컴파일: {STAN_WEAK}")
        model_weak = CmdStanModel(stan_file=STAN_WEAK)
        stan_data_w = make_stan_data_weak(y, gender)
        fit_weak = run_mcmc(model_weak, stan_data_w, "약한 사전 분포",
                            chains=args.chains, warmup=args.warmup,
                            sampling=args.sampling)
        df_sum_w, draws_weak = summarize_fit(fit_weak, "약한 사전 분포")
        all_results.append(df_sum_w)
        out_w = os.path.join(OUT_DIR, 'ss_mcmc_weakprior_N86.csv')
        draws_weak[[c for c in draws_weak.columns
                    if any(p in c for p in KEY_PARAMS + ['lp__', 'chain', 'iter'])]
                   ].to_csv(out_w, index=False)
        print(f"저장: {out_w}")
    else:
        draws_weak = None; df_sum_w = pd.DataFrame()

    # ── 모형 2: 강한 사전 분포 (sem_pcm_with_prior.stan) ──
    if not args.weak:
        print(f"\n모형 컴파일: {STAN_STRONG}")
        model_strong = CmdStanModel(stan_file=STAN_STRONG)
        stan_data_s = make_stan_data_strong(y, gender, priors)
        fit_strong = run_mcmc(model_strong, stan_data_s, "강한 사전 분포",
                              chains=args.chains, warmup=args.warmup,
                              sampling=args.sampling)
        df_sum_s, draws_strong = summarize_fit(fit_strong, "강한 사전 분포")
        all_results.append(df_sum_s)
        out_s = os.path.join(OUT_DIR, 'ss_mcmc_strongprior_N86.csv')
        draws_strong[[c for c in draws_strong.columns
                      if any(p in c for p in KEY_PARAMS + ['lp__', 'chain', 'iter'])]
                     ].to_csv(out_s, index=False)
        print(f"저장: {out_s}")
    else:
        draws_strong = None; df_sum_s = pd.DataFrame()

    # ── 비교 출력 ────────────────────────────────────────
    if draws_weak is not None and draws_strong is not None:
        print_comparison(df_sum_w, df_sum_s)
        bayesian_mediation_report(draws_weak, draws_strong)

    # ── 비교 CSV 저장 ────────────────────────────────────
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        comp_file = os.path.join(OUT_DIR, 'ss_prior_comparison_N86.csv')
        df_all.to_csv(comp_file, index=False)
        print(f"\n비교표 저장: {comp_file}")

    print("\n" + "=" * 60)
    print("완료. 다음 단계:")
    print("  결과 CSV 파일을 Claude에게 전달하면 논문 분석에 반영됩니다.")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='원논문 통계 기반 Bayesian PCM-SEM: 약한/강한 사전 분포 비교')
    parser.add_argument('--ppc',      action='store_true',
                        help='사전 예측 점검(Prior Predictive Check)만 실행')
    parser.add_argument('--weak',     action='store_true',
                        help='약한 사전 분포 모형만 실행')
    parser.add_argument('--strong',   action='store_true',
                        help='강한 사전 분포 모형만 실행')
    parser.add_argument('--chains',   type=int, default=CHAINS)
    parser.add_argument('--warmup',   type=int, default=ITER_WARMUP)
    parser.add_argument('--sampling', type=int, default=ITER_SAMPLING)
    args = parser.parse_args()

    for f in [STAN_WEAK, STAN_STRONG]:
        if not os.path.exists(f):
            print(f"ERROR: Stan 파일 없음: {f}")
            sys.exit(1)

    main(args)
