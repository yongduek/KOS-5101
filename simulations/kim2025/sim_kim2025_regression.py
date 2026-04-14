"""sim_kim2025_regression.py

Replicates the mediation analysis from:
  Kim et al. (2025). "중국유학생의 자아존중감과 문화적응스트레스의 관계:
  상호문화감수성의 매개역할"
  교육과 문화 (Journal of Education and Culture), 7(1), 254-267.
  DOI: 10.23009/JEC.2025.7.1.254

Analysis follows SPSS PROCESS Macro Model 4 (Simple Mediation):
  X  = Self-Esteem              (RSES,  자아존중감)
  M  = Intercultural Sensitivity (ISS,   상호문화감수성)
  Y  = Acculturative Stress     (RSSIS, 문화적응스트레스)
  Covariates: Gender, Academic_Year, TOPIK_Level, Economic_Status

Three regression equations (matching Table 3 in the paper):
  Model 1 (total effect)  : Y ~ X + covariates
  Model 2 (path a)        : M ~ X + covariates
  Model 3 (direct effect) : Y ~ X + M + covariates
Indirect effect = a × b, with 5 000-sample percentile bootstrap CI.

Dependencies: numpy, pandas only (no statsmodels / scipy required).
"""

import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal OLS engine (numpy only)
# ─────────────────────────────────────────────────────────────────────────────

def _erfc_pval(t_val: float) -> float:
    """Two-tailed p-value via normal approximation (accurate for df ≥ 100)."""
    return 2.0 * math.erfc(abs(t_val) / math.sqrt(2.0))


def _f_pval(f_stat: float, df1: int, df2: int) -> float:
    """
    F-distribution p-value via Wilson-Hilferty chi-squared approximation.
    Accurate for df2 ≥ 100.
    """
    chi2 = df1 * f_stat
    k    = float(df1)
    wh   = (chi2 / k) ** (1.0 / 3.0)
    mu   = 1.0 - 2.0 / (9.0 * k)
    sig  = math.sqrt(2.0 / (9.0 * k))
    z    = (wh - mu) / sig
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols_fit(y: np.ndarray, X: np.ndarray):
    """
    Ordinary Least Squares with automatic intercept.

    Parameters
    ----------
    y : (n,)   dependent variable
    X : (n, k) predictor matrix WITHOUT intercept column

    Returns
    -------
    coef   : (k+1,) coefficients, index 0 = intercept
    se     : (k+1,) standard errors
    t      : (k+1,) t-values
    p      : (k+1,) two-tailed p-values (normal approx)
    r2     : R²
    f_stat : F-statistic
    df1, df2 : numerator / denominator degrees of freedom for F
    f_p    : p-value for F-statistic
    """
    n   = len(y)
    Xc  = np.c_[np.ones(n), X]          # add intercept
    k   = Xc.shape[1]
    XtXi = np.linalg.inv(Xc.T @ Xc)
    coef  = XtXi @ Xc.T @ y
    resid = y - Xc @ coef
    s2    = (resid @ resid) / (n - k)
    se    = np.sqrt(s2 * np.diag(XtXi))
    t     = coef / se
    p     = np.array([_erfc_pval(ti) for ti in t])

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2  = 1.0 - ss_res / ss_tot
    df1 = k - 1
    df2 = n - k
    f_stat = (r2 / df1) / ((1.0 - r2) / df2)
    f_p    = _f_pval(f_stat, df1, df2)
    return coef, se, t, p, r2, f_stat, df1, df2, f_p


def std_betas(y: np.ndarray, X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Standardized regression coefficients (skip intercept at index 0)."""
    sy = y.std(ddof=1)
    sx = X.std(axis=0, ddof=1)
    return coef[1:] * sx / sy


def sig_star(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "   "


# ─────────────────────────────────────────────────────────────────────────────
#  Scale scoring (item sums with reversal)
# ─────────────────────────────────────────────────────────────────────────────

def compute_totals(df: pd.DataFrame):
    """Return (rses, rssis, iss) as float arrays of corrected scale totals."""
    rses_cols  = sorted(c for c in df.columns if c.startswith("RSES_"))
    rssis_cols = sorted(c for c in df.columns if c.startswith("RSSIS_"))
    iss_cols   = sorted(c for c in df.columns if c.startswith("ISS_"))

    rses  = df[rses_cols].apply(
        lambda row: sum(5 - row[c] if "REV" in c else row[c] for c in rses_cols), axis=1
    ).to_numpy(float)
    rssis = df[rssis_cols].sum(axis=1).to_numpy(float)
    iss   = df[iss_cols].apply(
        lambda row: sum(6 - row[c] if "REV" in c else row[c] for c in iss_cols), axis=1
    ).to_numpy(float)
    return rses, rssis, iss


# ─────────────────────────────────────────────────────────────────────────────
#  Paper reference values (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

_PAPER_COEF = {
    # (model_key, predictor) : (B, SE, β, t)
    ("m1", "RSES"):            (-2.72,  .27, -.45, -10.12),
    ("m1", "Gender"):          (-16.25, 2.89, -.26,  -5.63),
    ("m1", "Academic_Year"):   (-1.76,  1.53, -.05,  -1.15),
    ("m1", "TOPIK_Level"):     (-4.34,  4.35, -.05,  -1.00),
    ("m1", "Economic_Status"): ( 3.35,  2.91,  .05,   1.15),

    ("m2", "RSES"):            ( 1.15,  .11,  .49,  10.96),
    ("m2", "Gender"):          ( 3.72,  1.13,  .15,   3.30),
    ("m2", "Academic_Year"):   (  .68,   .60,  .05,   1.14),
    ("m2", "TOPIK_Level"):     ( 2.26,  1.70,  .06,   1.33),
    ("m2", "Economic_Status"): (-1.01,  1.14, -.04,   -.89),

    ("m3", "RSES"):            (-1.76,   .29, -.29,  -6.03),
    ("m3", "ISS"):             ( -.84,   .12, -.33,  -6.81),
    ("m3", "Gender"):          (-13.14, 2.77, -.21,  -4.74),
    ("m3", "Academic_Year"):   (-1.19,  1.45, -.04,   -.82),
    ("m3", "TOPIK_Level"):     (-2.44,  4.12, -.03,   -.59),
    ("m3", "Economic_Status"): ( 2.50,  2.76,  .04,    .91),
}

_PAPER_MODEL = {
    "m1": {"r2": .26, "f": 27.35, "df": "(5,388)"},
    "m2": {"r2": .26, "f": 26.73, "df": "(5,388)"},
    "m3": {"r2": .34, "f": 33.18, "df": "(6,387)"},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Formatted regression block (matching Table 3 layout)
# ─────────────────────────────────────────────────────────────────────────────

def print_model(title: str,
                y:     np.ndarray,
                X:     np.ndarray,
                names: list,
                mkey:  str) -> tuple:
    """Print one regression block; return (coef, se)."""
    coef, se, t, p, r2, f, df1, df2, f_p = ols_fit(y, X)
    beta = std_betas(y, X, coef)
    pm   = _PAPER_MODEL[mkey]

    print(f"\n{'─'*82}")
    print(f"  {title}")
    print(f"{'─'*82}")
    print(f"  {'Predictor':<20}  {'B':>8}  {'SE':>6}  {'β':>7}  {'t':>8}  {'':3}  "
          f"paper B / β / t")
    print(f"  {'─'*79}")

    for i, name in enumerate(names):
        B_    = coef[i + 1]
        SE_   = se  [i + 1]
        beta_ = beta[i]
        t_    = t   [i + 1]
        p_    = p   [i + 1]
        star  = sig_star(p_)
        ref   = _PAPER_COEF.get((mkey, name))
        pstr  = (f"B={ref[0]:+.2f} / β={ref[2]:+.2f} / t={ref[3]:+.2f}"
                 if ref else "—")
        print(f"  {name:<20}  {B_:>8.2f}  {SE_:>6.2f}  {beta_:>+7.2f}  {t_:>8.2f}  "
              f"{star}  [{pstr}]")

    f_star = sig_star(f_p)
    print(f"\n  R²={r2:.3f} [paper {pm['r2']:.2f}]   "
          f"F({df1},{df2})={f:.2f}{f_star}  "
          f"[paper F{pm['df']}={pm['f']:.2f}***]")
    return coef, se


# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap indirect effect  (PROCESS-style percentile CI)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_indirect(rses:  np.ndarray,
                       rssis: np.ndarray,
                       iss:   np.ndarray,
                       cov:   np.ndarray,
                       n_boot: int = 5000,
                       seed:   int = 42) -> tuple:
    """
    Percentile bootstrap 95% CI for the indirect effect a × b.
    Returns (point_estimate, lower_CI, upper_CI).
    """
    rng      = np.random.default_rng(seed)
    n        = len(rses)
    indirect = np.empty(n_boot)

    for i in range(n_boot):
        idx  = rng.integers(0, n, n)
        rb, rsb, isb, cb = rses[idx], rssis[idx], iss[idx], cov[idx]

        # path a : ISS ~ RSES + covariates
        Xa = np.c_[np.ones(n), rb, cb]
        a  = np.linalg.lstsq(Xa, isb, rcond=None)[0][1]

        # path b : RSSIS ~ RSES + ISS + covariates
        Xb = np.c_[np.ones(n), rb, isb, cb]
        b  = np.linalg.lstsq(Xb, rsb, rcond=None)[0][2]

        indirect[i] = a * b

    pt = float(indirect.mean())
    lo = float(np.percentile(indirect, 2.5))
    hi = float(np.percentile(indirect, 97.5))
    return pt, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("data_combined_simulated.csv")
    rses, rssis, iss = compute_totals(df)

    gender = df["Gender"].to_numpy(float)
    year   = df["Academic_Year"].to_numpy(float)
    topik  = df["TOPIK_Level"].to_numpy(float)
    eco    = df["Economic_Status"].to_numpy(float)
    cov    = np.c_[gender, year, topik, eco]
    cov_names = ["Gender", "Academic_Year", "TOPIK_Level", "Economic_Status"]

    print("=" * 82)
    print("  Kim et al. (2025)  —  Mediation Analysis Replication")
    print("  PROCESS Macro Model 4 (Simple Mediation, N=394)")
    print("  X=RSES (Self-Esteem)  M=ISS (ICS)  Y=RSSIS (Accult. Stress)")
    print("=" * 82)

    # ── Model 1 : total effect  Y ~ X + covariates ───────────────────────────
    X1      = np.c_[rses, cov]
    c1, s1  = print_model(
        "Model 1 – Total Effect   DV = Acculturative Stress (RSSIS)",
        rssis, X1, ["RSES"] + cov_names, "m1")

    # ── Model 2 : path a  M ~ X + covariates ─────────────────────────────────
    X2      = np.c_[rses, cov]
    c2, s2  = print_model(
        "Model 2 – Path a         DV = Intercultural Sensitivity (ISS)",
        iss, X2, ["RSES"] + cov_names, "m2")

    # ── Model 3 : direct effect  Y ~ X + M + covariates ──────────────────────
    X3      = np.c_[rses, iss, cov]
    c3, s3  = print_model(
        "Model 3 – Direct Effect  DV = Acculturative Stress (RSSIS)",
        rssis, X3, ["RSES", "ISS"] + cov_names, "m3")

    # ── Mediation effects summary ─────────────────────────────────────────────
    B_tot, SE_tot = c1[1], s1[1]
    B_dir, SE_dir = c3[1], s3[1]
    a_coef        = c2[1]   # path a : RSES → ISS
    b_coef        = c3[2]   # path b : ISS  → RSSIS  (controlling for RSES)

    z = 1.960   # 95% CI multiplier (large-sample)
    tot_lo, tot_hi = B_tot - z * SE_tot, B_tot + z * SE_tot
    dir_lo, dir_hi = B_dir - z * SE_dir, B_dir + z * SE_dir

    print(f"\n  Bootstrapping indirect effect ({5000:,} samples) …", flush=True)
    B_ind, ind_lo, ind_hi = bootstrap_indirect(rses, rssis, iss, cov)

    print(f"\n{'─'*82}")
    print("  Mediation Effects Summary")
    print(f"{'─'*82}")
    print(f"  Total  effect  B = {B_tot:+.2f}  SE={SE_tot:.2f}  "
          f"95% CI [{tot_lo:.2f}, {tot_hi:.2f}]")
    print(f"                 paper: B=-2.72  SE=0.27  95% CI [-3.25, -2.19]")
    print(f"  Direct effect  B = {B_dir:+.2f}  SE={SE_dir:.2f}  "
          f"95% CI [{dir_lo:.2f}, {dir_hi:.2f}]")
    print(f"                 paper: B=-1.76  SE=0.29  95% CI [-2.33, -1.18]")
    print(f"  Indirect a×b   B = {B_ind:+.2f}  "
          f"(a={a_coef:.3f} × b={b_coef:.3f} = {a_coef*b_coef:.3f})")
    print(f"  Bootstrap CI   95% CI [{ind_lo:.2f}, {ind_hi:.2f}]  (5,000 samples)")
    print(f"                 paper: B=-0.97  95% CI [-1.31, -0.66]")

    excl_zero = (ind_lo * ind_hi) > 0
    verdict = ("✓ CI excludes 0 → statistically significant partial mediation"
               if excl_zero else "✗ CI includes 0 → mediation not confirmed")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
