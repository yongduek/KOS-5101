#!/usr/bin/env python3
"""
김연하 외 (2025) — 문항 수준 시뮬레이션 데이터 생성기
================================================================
논문: 중국유학생의 자아존중감과 문화적응스트레스의 관계:
      상호문화감수성의 매개역할. 교육과 문화, 7(1), 254-267.

출력: 실제 설문과 동일한 문항별 Likert 응답값
  공변량(4) + RSES 10문항(1-4점) + ISS 24문항(1-5점) + ASSIS 36문항(1-5점)
  = 74열 × 394행

핵심 알고리즘: 1-factor CFA 역산 시뮬레이션
  ① 논문 기술통계 + Cronbach α → 문항 파라미터(λ, σ_unique)
  ② 구조방정식으로 척도 간 인과관계 반영(X→M, X+M→Y)
  ③ 잠재요인점수 θ = (합산-k·μ_item)/(k·λ)  [표준화 단위]
  ④ item_ij = μ_item + λ·θ_i + ε_ij         [ε ~ N(0, σ_unique)]
  ⑤ 반올림 → Likert 범위 클리핑
================================================================
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 0. 재현성 시드
# ══════════════════════════════════════════════════════════════════════════════
SEED = 2025
rng  = np.random.default_rng(SEED)
N    = 394

# ══════════════════════════════════════════════════════════════════════════════
# 1. 논문 모수 (표 2, 표 3)
# ══════════════════════════════════════════════════════════════════════════════
# 척도 정보: (문항수, Likert최대점수, 합산평균, 합산SD, Cronbach_α)
SCALE = {
    "X": dict(k=10, pts=4, mu=29.87, sd= 5.24, alpha=0.84),  # 자아존중감 RSES
    "M": dict(k=24, pts=5, mu=86.10, sd=12.27, alpha=0.81),  # 상호문화감수성 ISS
    "Y": dict(k=36, pts=5, mu=81.84, sd=31.49, alpha=0.97),  # 문화적응스트레스 ASSIS
}

# 회귀계수 (Step 2: X→M)
B2 = dict(X=1.15, gender=3.72, grade=0.68, topik=2.26, income=-1.01)
R2_M = 0.26

# 회귀계수 (Step 3: X+M→Y)
B3 = dict(X=-1.76, M=-0.84, gender=-13.14, grade=-1.19, topik=-2.44, income=2.50)
R2_Y = 0.34

# ══════════════════════════════════════════════════════════════════════════════
# 2. 문항 파라미터 역산 함수
# ══════════════════════════════════════════════════════════════════════════════
def item_params_from_alpha(k, sd_total, alpha):
    """
    1-factor CFA 파라미터를 논문 보고값으로부터 역산.

    Returns
    -------
    mu_item  : 평균 문항 점수 (= μ_total / k)
    lam      : 공통요인 부하량 (σ_item 단위)
    sd_u     : 문항 고유 잔차 SD
    r        : 평균 문항 간 상관
    sd_item  : 문항 표준편차
    """
    r = alpha / (k - (k - 1) * alpha)     # 평균 문항 간 상관
    var_item = sd_total**2 / (k * (1 + (k - 1) * r))
    sd_item  = np.sqrt(var_item)
    lam      = np.sqrt(r) * sd_item       # factor loading (σ_item 단위)
    sd_u     = sd_item * np.sqrt(1 - r)   # unique SD
    return r, sd_item, lam, sd_u


# ══════════════════════════════════════════════════════════════════════════════
# 3. 문항 생성 함수
# ══════════════════════════════════════════════════════════════════════════════
def gen_items(total_scores, k, pts, mu_item, lam, sd_u, rng):
    """
    합산점수 → 문항별 Likert 응답 생성 (1-factor CFA 역산).

    핵심 수식
    ---------
    θ_i  = (total_i - k·μ_item) / (k·λ)   ← 잠재요인점수 (표준화)
    x_ij = μ_item + λ·θ_i + ε_ij           ← 문항 응답 (연속)
    반올림 후 [1, pts] 클리핑

    결과
    ----
    items : (N, k) 정수 배열
    """
    N   = len(total_scores)
    theta = (total_scores - k * mu_item) / (k * lam)   # (N,) 표준화 잠재요인
    common = (lam * theta)[:, None] * np.ones((1, k))  # (N, k) 공통성분
    unique = rng.normal(0, sd_u, (N, k))               # (N, k) 고유성분
    raw    = mu_item + common + unique
    return np.clip(np.round(raw).astype(int), 1, pts)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 공변량 생성 (논문 표 1)
# ══════════════════════════════════════════════════════════════════════════════
gender = rng.choice([1, 2],       N, p=[0.477, 0.523])
grade  = rng.choice([1, 2, 3, 4], N, p=[0.096, 0.404, 0.256, 0.244])
topik  = rng.choice([0, 1],       N, p=[0.124, 0.876])
income = rng.choice([1, 2, 3],    N, p=[0.063, 0.736, 0.201])

# ══════════════════════════════════════════════════════════════════════════════
# 5. 합산점수 생성 (구조방정식 — 이전 검증된 방식)
# ══════════════════════════════════════════════════════════════════════════════
# ── X 합산점수 ────────────────────────────────────────────────────────────────
mu_X, sd_X = SCALE["X"]["mu"], SCALE["X"]["sd"]
X_total = rng.normal(mu_X, sd_X, N)
X_total = (X_total - X_total.mean()) / X_total.std() * sd_X + mu_X
X_total = np.clip(X_total, SCALE["X"]["k"] * 1, SCALE["X"]["k"] * SCALE["X"]["pts"])

# ── M 합산점수 (Step 2 구조방정식) ───────────────────────────────────────────
sd_eps_M = np.sqrt((1 - R2_M) * SCALE["M"]["sd"]**2)
eps_M    = rng.normal(0, sd_eps_M, N)
M_total  = (B2["X"]      * X_total
          + B2["gender"] * gender
          + B2["grade"]  * grade
          + B2["topik"]  * topik
          + B2["income"] * income
          + eps_M)
M_total += SCALE["M"]["mu"] - M_total.mean()
M_total  = np.clip(M_total, SCALE["M"]["k"] * 1, SCALE["M"]["k"] * SCALE["M"]["pts"])

# ── Y 합산점수 (Step 3 구조방정식) ───────────────────────────────────────────
sd_eps_Y = np.sqrt((1 - R2_Y) * SCALE["Y"]["sd"]**2)
eps_Y    = rng.normal(0, sd_eps_Y, N)
Y_total  = (B3["X"]      * X_total
          + B3["M"]      * M_total
          + B3["gender"] * gender
          + B3["grade"]  * grade
          + B3["topik"]  * topik
          + B3["income"] * income
          + eps_Y)
Y_total += SCALE["Y"]["mu"] - Y_total.mean()
Y_total  = np.clip(Y_total, SCALE["Y"]["k"] * 1, SCALE["Y"]["k"] * SCALE["Y"]["pts"])

# ══════════════════════════════════════════════════════════════════════════════
# 6. 문항 파라미터 계산 후 문항 생성
# ══════════════════════════════════════════════════════════════════════════════
params = {}
for key, ts in [("X", X_total), ("M", M_total), ("Y", Y_total)]:
    s = SCALE[key]
    r, sd_i, lam, sd_u = item_params_from_alpha(s["k"], ts.std(), s["alpha"])
    params[key] = dict(mu_item=ts.mean()/s["k"], lam=lam, sd_u=sd_u,
                        r=r, sd_item=sd_i)

print("─── 문항 파라미터 ───────────────────────────────────────────────────")
for key in ("X","M","Y"):
    p = params[key]; s = SCALE[key]
    print(f"[{key}] k={s['k']}, pts={s['pts']}, "
          f"μ_item={p['mu_item']:.3f}, r={p['r']:.3f}, "
          f"λ={p['lam']:.3f}, σ_unique={p['sd_u']:.3f}")

print("\n문항 생성 중...")
p = params["X"]; s = SCALE["X"]
items_X = gen_items(X_total, s["k"], s["pts"],
                    p["mu_item"], p["lam"], p["sd_u"], rng)

p = params["M"]; s = SCALE["M"]
items_M = gen_items(M_total, s["k"], s["pts"],
                    p["mu_item"], p["lam"], p["sd_u"], rng)

p = params["Y"]; s = SCALE["Y"]
items_Y = gen_items(Y_total, s["k"], s["pts"],
                    p["mu_item"], p["lam"], p["sd_u"], rng)

# 문항합 = 분석에 사용할 실제 합산점수
sum_X = items_X.sum(axis=1).astype(float)
sum_M = items_M.sum(axis=1).astype(float)
sum_Y = items_Y.sum(axis=1).astype(float)

# ══════════════════════════════════════════════════════════════════════════════
# 7. 검증
# ══════════════════════════════════════════════════════════════════════════════
def ols(Xm, y):
    c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    r2 = 1 - np.sum((y - Xm@c)**2) / np.sum((y - y.mean())**2)
    return c, r2

def dm(*arrays):
    return np.column_stack([np.ones(N)] + list(arrays))

def cronbach(mat):
    k = mat.shape[1]
    return k/(k-1) * (1 - mat.var(axis=0,ddof=1).sum() /
                          mat.sum(axis=1).var(ddof=1))

sep = "─" * 65
print("\n" + "═"*65)
print("  검증 보고서")
print("═"*65)

# ① 기술통계
print(f"\n{sep}\n  ① 기술통계 (문항합 기준)\n{sep}")
print(f"  {'척도':<20} {'논문 μ':>9} {'시뮬 μ':>9}   {'논문 σ':>8} {'시뮬 σ':>8}")
for arr, label, pm, ps in [
    (sum_X, "자아존중감",       29.87,  5.24),
    (sum_M, "상호문화감수성",   86.10, 12.27),
    (sum_Y, "문화적응스트레스", 81.84, 31.49),
]:
    sm=arr.mean(); ss=arr.std()
    fm="✓" if abs(sm-pm)<2 else "△"
    fs="✓" if abs(ss-ps)<2 else "△"
    print(f"  {label:<20} {pm:>9.2f} {sm:>9.2f}{fm}   {ps:>8.2f} {ss:>8.2f}{fs}")

# ② 상관관계
print(f"\n{sep}\n  ② 척도 간 상관계수\n{sep}")
for lbl, pv, a, b in [
    ("r(X,M)", 0.46, sum_X, sum_M),
    ("r(X,Y)",-0.41, sum_X, sum_Y),
    ("r(M,Y)",-0.49, sum_M, sum_Y),
]:
    sv = np.corrcoef(a, b)[0,1]
    d  = abs(sv-pv)
    print(f"  {lbl:<10} 논문={pv:>6.3f}  시뮬={sv:>6.3f}  |차이|={d:.4f} "
          f"{'✓' if d<0.08 else '△'}")

# ③ Cronbach's α
print(f"\n{sep}\n  ③ Cronbach's α\n{sep}")
for mat, lbl, pa in [
    (items_X, "RSES(자아존중감)",        0.84),
    (items_M, "ISS(상호문화감수성)",      0.81),
    (items_Y, "ASSIS(문화적응스트레스)",  0.97),
]:
    sa = cronbach(mat)
    d  = abs(sa-pa)
    print(f"  {lbl:<30} 논문={pa:.3f}  시뮬={sa:.3f}  |차이|={d:.4f} "
          f"{'✓' if d<0.06 else '△'}")

# ④ 매개분석
print(f"\n{sep}\n  ④ 매개분석 (문항합 기준)\n{sep}")
COV = [gender, grade, topik, income]
c2, r2_m = ols(dm(sum_X,*COV), sum_M)
c3, r2_y = ols(dm(sum_X, sum_M,*COV), sum_Y)
c1, _    = ols(dm(sum_X,*COV), sum_Y)
a_sim=c2[1]; b_sim=c3[2]; dir_sim=c3[1]; tot_sim=c1[1]; ind_sim=a_sim*b_sim

print(f"  {'효과':<28} {'논문 B':>9}  {'시뮬 B':>9}  {'|차이|':>7}")
for lbl,pv,sv in [
    ("a경로 (X→M)",         1.15, a_sim),
    ("b경로 (M→Y)",        -0.84, b_sim),
    ("직접효과 c'",         -1.76, dir_sim),
    ("총효과 c",            -2.72, tot_sim),
    ("간접효과 a×b",        -0.97, ind_sim),
    ("R²(M)",                0.26, r2_m),
    ("R²(Y|X,M)",            0.34, r2_y),
]:
    d=abs(sv-pv); f="✓" if d<0.5 else "△"
    print(f"  {lbl:<28} {pv:>9.3f}  {sv:>9.3f}  {d:>7.4f} {f}")

# ⑤ 문항 분포 샘플
print(f"\n{sep}\n  ⑤ 문항 응답 분포 샘플 (첫 3문항)\n{sep}")
for mat, pfx, pts in [(items_X,"RSES",4),(items_M,"ISS",5),(items_Y,"ASSIS",5)]:
    print(f"  [{pfx}]  점수 범위: 1–{pts}")
    for j in range(3):
        vals,cnts = np.unique(mat[:,j], return_counts=True)
        dist = "  ".join(f"{v}점:{c/N*100:.0f}%" for v,c in zip(vals,cnts))
        print(f"    문항{j+1}: {dist}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. DataFrame 저장
# ══════════════════════════════════════════════════════════════════════════════
rses_cols  = [f"rses_{i+1}"  for i in range(10)]
iss_cols   = [f"iss_{i+1}"   for i in range(24)]
assis_cols = [f"assis_{i+1}" for i in range(36)]

df = pd.concat([
    pd.DataFrame({"gender":gender,"grade":grade,"topik":topik,"income":income}),
    pd.DataFrame(items_X, columns=rses_cols),
    pd.DataFrame(items_M, columns=iss_cols),
    pd.DataFrame(items_Y, columns=assis_cols),
], axis=1)

# 참조용 합산열
df["total_rses"]  = sum_X
df["total_iss"]   = sum_M
df["total_assis"] = sum_Y

# 파일 1: 문항 수준 (메인)
out1 = "/sessions/eloquent-inspiring-faraday/mnt/shim/kim2025_item_data.csv"
df.to_csv(out1, index=False, encoding="utf-8-sig")

# 파일 2: 합산점수만 (Stan 매개분석용)
df2 = df[["gender","grade","topik","income",
          "total_rses","total_iss","total_assis"]].copy()
df2.columns = ["gender","grade","topik","income",
               "self_esteem","interc_sensitivity","acculturative_stress"]
out2 = "/sessions/eloquent-inspiring-faraday/mnt/shim/kim2025_total_data.csv"
df2.to_csv(out2, index=False, encoding="utf-8-sig")

print(f"\n{'═'*65}")
print(f"  저장 완료")
print(f"  [문항수준] kim2025_item_data.csv    {len(df)}행 × {len(df.columns)}열")
print(f"  [합산점수] kim2025_total_data.csv   {len(df2)}행 × {len(df2.columns)}열")
print(f"{'═'*65}")
print(f"""
  열 구조 (kim2025_item_data.csv, 총 {len(df.columns)}열)
  ─────────────────────────────────────────────────────────
  공변량(4)     gender, grade, topik, income
  RSES(10)      rses_1 ~ rses_10       (자아존중감,     1–4점)
  ISS(24)       iss_1  ~ iss_24        (상호문화감수성,  1–5점)
  ASSIS(36)     assis_1 ~ assis_36     (문화적응스트레스, 1–5점)
  합산참조(3)   total_rses, total_iss, total_assis
  ─────────────────────────────────────────────────────────
  gender : 1=남(male), 2=여(female)
  grade  : 1~4 (학년)
  topik  : 0=초급(TOPIK 1-2급), 1=중고급(3-6급)
  income : 1=하, 2=중, 3=상
""")
