#!/usr/bin/env python3
"""
김연하 외 (2025) 논문 시뮬레이션 데이터 생성기
================================================================
논문: 중국유학생의 자아존중감과 문화적응스트레스의 관계:
      상호문화감수성의 매개역할
출처: 교육과 문화, 7(1), 254-267.

논문에 보고된 기술통계·상관계수·회귀계수·R² 값을 역산하여
원논문과 동일한 통계 구조를 갖는 시뮬레이션 데이터를 생성합니다.

의존 패키지: numpy, pandas (표준 설치)
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼: numpy 기반 OLS 회귀
# ══════════════════════════════════════════════════════════════════════════════
def ols(X_mat, y):
    """
    OLS 회귀: X_mat (N x p 설계행렬, 절편 포함), y (N,)
    반환: coef, r_squared, residuals
    """
    coef, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    y_hat    = X_mat @ coef
    ss_res   = np.sum((y - y_hat) ** 2)
    ss_tot   = np.sum((y - y.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot
    return coef, r2, y - y_hat

def design(df_local, predictors):
    """절편을 포함한 설계행렬 반환"""
    X_mat = np.column_stack([np.ones(len(df_local))]
                            + [df_local[p].values for p in predictors])
    return X_mat


# ══════════════════════════════════════════════════════════════════════════════
# 0. 재현성 시드
# ══════════════════════════════════════════════════════════════════════════════
SEED = 2025
rng  = np.random.default_rng(SEED)
N    = 394

# ══════════════════════════════════════════════════════════════════════════════
# 1. 논문에서 추출한 모수 (논문 표 2, 표 3)
# ══════════════════════════════════════════════════════════════════════════════
# (평균, 표준편차, 최솟값, 최댓값)
STATS = {
    "X": (29.87,  5.24,  10,  40),   # 자아존중감 (RSES)
    "M": (86.10, 12.27,  24, 120),   # 상호문화감수성 (ISS)
    "Y": (81.84, 31.49,  36, 180),   # 문화적응스트레스 (ASSIS)
}

# 상관계수 (표 2)
R_XM, R_XY, R_MY =  0.46, -0.41, -0.49

# Step 2: X → M  (비표준화 B)
B2 = {"X": 1.15, "gender": 3.72, "grade": 0.68, "topik": 2.26, "income": -1.01}
R2_M = 0.26

# Step 3: X + M → Y  (비표준화 B)
B3 = {"X": -1.76, "M": -0.84,
      "gender": -13.14, "grade": -1.19, "topik": -2.44, "income": 2.50}
R2_Y = 0.34

# ══════════════════════════════════════════════════════════════════════════════
# 2. 공변량 생성 (논문 표 1 빈도 기반)
# ══════════════════════════════════════════════════════════════════════════════
gender = rng.choice([1, 2],       N, p=[0.477, 0.523])
grade  = rng.choice([1, 2, 3, 4], N, p=[0.096, 0.404, 0.256, 0.244])
topik  = rng.choice([0, 1],       N, p=[0.124, 0.876])
income = rng.choice([1, 2, 3],    N, p=[0.063, 0.736, 0.201])

# ══════════════════════════════════════════════════════════════════════════════
# 3. X (자아존중감) 생성
# ══════════════════════════════════════════════════════════════════════════════
mu_X, sd_X, lo_X, hi_X = STATS["X"]
X_raw = rng.normal(mu_X, sd_X, N)
# moment matching: 정확한 평균·SD 부여
X_raw = (X_raw - X_raw.mean()) / X_raw.std() * sd_X + mu_X
X = np.clip(X_raw, lo_X, hi_X)

# ══════════════════════════════════════════════════════════════════════════════
# 4. M (상호문화감수성) 생성 — 구조방정식
# ══════════════════════════════════════════════════════════════════════════════
# M = B_X*X + B_gender*gender + ... + ε_M
# SD(ε_M) = sqrt( (1 - R²_M) * Var(M) )
mu_M, sd_M, lo_M, hi_M = STATS["M"]
sd_eps_M = np.sqrt((1 - R2_M) * sd_M**2)          # ≈ 10.56

eps_M = rng.normal(0, sd_eps_M, N)
M_raw = (B2["X"]      * X
       + B2["gender"] * gender
       + B2["grade"]  * grade
       + B2["topik"]  * topik
       + B2["income"] * income
       + eps_M)
M_raw += (mu_M - M_raw.mean())                     # 절편 보정
M = np.clip(M_raw, lo_M, hi_M)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Y (문화적응스트레스) 생성 — 구조방정식
# ══════════════════════════════════════════════════════════════════════════════
# SD(ε_Y) = sqrt( (1 - R²_Y) * Var(Y) )
mu_Y, sd_Y, lo_Y, hi_Y = STATS["Y"]
sd_eps_Y = np.sqrt((1 - R2_Y) * sd_Y**2)          # ≈ 25.58

eps_Y = rng.normal(0, sd_eps_Y, N)
Y_raw = (B3["X"]      * X
       + B3["M"]      * M
       + B3["gender"] * gender
       + B3["grade"]  * grade
       + B3["topik"]  * topik
       + B3["income"] * income
       + eps_Y)
Y_raw += (mu_Y - Y_raw.mean())                     # 절편 보정
Y = np.clip(Y_raw, lo_Y, hi_Y)

# ══════════════════════════════════════════════════════════════════════════════
# 6. DataFrame 생성
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame({
    "gender":               gender,
    "grade":                grade,
    "topik":                topik,
    "income":               income,
    "self_esteem":          X,
    "interc_sensitivity":   M,
    "acculturative_stress": Y,
})

# ══════════════════════════════════════════════════════════════════════════════
# 7. 검증 (numpy OLS)
# ══════════════════════════════════════════════════════════════════════════════
COV = ["gender", "grade", "topik", "income"]
sep = "─" * 62

def check(label, paper, sim, tol):
    d    = abs(sim - paper)
    flag = "✓" if d <= tol else "△"
    print(f"  {label:<28} {paper:>9.3f}  {sim:>9.3f}  {d:>7.4f} {flag}")

print("\n" + "═"*62)
print("  김연하 외(2025) 시뮬레이션 검증 보고서")
print("═"*62)

# ① 기술통계
print(f"\n{sep}\n  ① 기술통계 (평균 / 표준편차)\n{sep}")
print(f"  {'변수':<22} {'논문 μ':>8} {'시뮬 μ':>8}  {'논문 σ':>8} {'시뮬 σ':>8}")
for col, label, pm, ps in [
    ("self_esteem",          "자아존중감",       29.87,  5.24),
    ("interc_sensitivity",   "상호문화감수성",   86.10, 12.27),
    ("acculturative_stress", "문화적응스트레스", 81.84, 31.49),
]:
    sm = df[col].mean(); ss = df[col].std()
    fm = "✓" if abs(sm-pm)<1.0 else "△"
    fs = "✓" if abs(ss-ps)<1.5 else "△"
    print(f"  {label:<22} {pm:>8.2f} {sm:>8.2f}{fm}  {ps:>8.2f} {ss:>8.2f}{fs}")

# ② 상관관계
print(f"\n{sep}\n  ② 변수 간 상관계수\n{sep}")
Xv = df["self_esteem"].values
Mv = df["interc_sensitivity"].values
Yv = df["acculturative_stress"].values

def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]

print(f"  {'쌍':<14} {'논문':>8} {'시뮬':>8}  {'|차이|':>7}")
for lbl, pv, sv in [
    ("r(X, M)",  0.46,  pearson(Xv, Mv)),
    ("r(X, Y)", -0.41,  pearson(Xv, Yv)),
    ("r(M, Y)", -0.49,  pearson(Mv, Yv)),
]:
    d = abs(sv - pv)
    f = "✓" if d < 0.05 else "△"
    print(f"  {lbl:<14} {pv:>8.3f} {sv:>8.3f}  {d:>7.4f} {f}")

# ③ Step 2: X → M
pred2  = ["self_esteem"] + COV
Xm2    = design(df, pred2)
coef2, r2_m_sim, _ = ols(Xm2, Mv)
names2 = ["절편"] + pred2

print(f"\n{sep}\n  ③ Step 2: X → M  (R²: 논문=.26, 시뮬={r2_m_sim:.3f})\n{sep}")
print(f"  {'계수':<28} {'논문 B':>9}  {'시뮬 B':>9}  {'|차이|':>7}")
ref2 = [None, B2["X"], B2["gender"], B2["grade"], B2["topik"], B2["income"]]
for nm, pv, sv in zip(names2, ref2, coef2):
    if pv is None: continue
    check(nm, pv, sv, 0.4)

# ④ Step 3: X + M → Y
pred3  = ["self_esteem", "interc_sensitivity"] + COV
Xm3    = design(df, pred3)
coef3, r2_y_sim, _ = ols(Xm3, Yv)
names3 = ["절편"] + pred3

print(f"\n{sep}\n  ④ Step 3: X + M → Y  (R²: 논문=.34, 시뮬={r2_y_sim:.3f})\n{sep}")
print(f"  {'계수':<28} {'논문 B':>9}  {'시뮬 B':>9}  {'|차이|':>7}")
ref3 = [None, B3["X"], B3["M"],
        B3["gender"], B3["grade"], B3["topik"], B3["income"]]
for nm, pv, sv in zip(names3, ref3, coef3):
    if pv is None: continue
    check(nm, pv, sv, 1.5)

# ⑤ 매개효과 분해
a_sim        = coef2[1]              # B_X in step 2
b_sim        = coef3[2]              # B_M in step 3
indirect_sim = a_sim * b_sim
direct_sim   = coef3[1]              # B_X in step 3
# 총효과: step1
Xm1 = design(df, ["self_esteem"] + COV)
coef1, r2_y1, _ = ols(Xm1, Yv)
total_sim = coef1[1]

print(f"\n{sep}\n  ⑤ 매개효과 분해\n{sep}")
print(f"  {'효과':<28} {'논문 B':>9}  {'시뮬 B':>9}  {'|차이|':>7}")
check("총효과",       -2.72, total_sim,    0.5)
check("직접효과",     -1.76, direct_sim,   0.5)
check("간접효과(a×b)",-0.97, indirect_sim, 0.3)

# ⑥ 부트스트랩 간접효과 (2,000회)
print(f"\n{sep}\n  ⑥ 간접효과 부트스트랩 95% CI (2,000회)\n{sep}")
boot = []
idx_all = np.arange(N)
df_arr = df.values   # numpy array for speed
cols   = list(df.columns)
col_X  = cols.index("self_esteem")
col_M  = cols.index("interc_sensitivity")
col_Y  = cols.index("acculturative_stress")

for _ in range(2000):
    idx = rng.choice(idx_all, N, replace=True)
    sample = df_arr[idx]
    Xv_b = sample[:, col_X]
    Mv_b = sample[:, col_M]
    Yv_b = sample[:, col_Y]
    cov_b = sample[:, [cols.index(c) for c in COV]]
    one_b = np.ones(N)

    # M ~ X + covariates
    Xm2b = np.column_stack([one_b, Xv_b] + [cov_b[:, i] for i in range(4)])
    c2b, *_ = np.linalg.lstsq(Xm2b, Mv_b, rcond=None)
    # Y ~ X + M + covariates
    Xm3b = np.column_stack([one_b, Xv_b, Mv_b] + [cov_b[:, i] for i in range(4)])
    c3b, *_ = np.linalg.lstsq(Xm3b, Yv_b, rcond=None)
    boot.append(c2b[1] * c3b[2])

ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
zero_in = ci_lo <= 0 <= ci_hi
print(f"  논문 CI: [-1.31, -0.66]")
print(f"  시뮬 CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  0 포함?: {'예 (비유의)' if zero_in else '아니오 → 매개효과 유의 ✓'}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 저장
# ══════════════════════════════════════════════════════════════════════════════
out_csv = "/sessions/eloquent-inspiring-faraday/mnt/shim/kim2025_simulated_data.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"\n{'═'*62}")
print(f"  저장 완료: kim2025_simulated_data.csv")
print(f"  행: {len(df)}  열: {len(df.columns)}  {list(df.columns)}")
print(f"{'═'*62}")
print("""
  변수 코딩
  gender  : 1=남(male), 2=여(female)
  grade   : 1~4 (학년)
  topik   : 0=초급, 1=중고급
  income  : 1=하, 2=중, 3=상
  self_esteem          : 자아존중감 합산 (RSES, 10~40)
  interc_sensitivity   : 상호문화감수성 합산 (ISS, 24~120)
  acculturative_stress : 문화적응스트레스 합산 (ASSIS, 36~180)
""")
