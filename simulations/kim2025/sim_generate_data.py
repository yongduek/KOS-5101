import numpy as np
import pandas as pd
import random
import os

def generate_items_for_total(total_score, num_items, min_val, max_val):
    # Ensure total_score is within possible bounds
    total_score = max(num_items * min_val, min(num_items * max_val, int(round(total_score))))

    # Initialize all items with the minimum value
    items = [min_val] * num_items
    current_total = sum(items)

    # Distribute the remaining score randomly across items
    remaining = total_score - current_total
    while remaining > 0:
        idx = random.randint(0, num_items - 1)
        if items[idx] < max_val:
            items[idx] += 1
            remaining -= 1

    # Swap +1/-1 pairs to add "natural" spread without changing the total sum
    for _ in range(num_items * 2):
        i, j = random.sample(range(num_items), 2)
        if items[i] < max_val and items[j] > min_val:
            items[i] += 1
            items[j] -= 1

    return items


def calibrate_integer_scores(latent_col, target_mean, target_sd, valid_min, valid_max, n_iter=60):
    """
    Iteratively rescale and shift the latent continuous scores so that,
    after clipping to [valid_min, valid_max] and rounding to integers,
    the resulting distribution matches target_mean and target_sd as closely
    as possible.
    """
    scores = latent_col.copy()

    for _ in range(n_iter):
        int_scores = np.clip(np.round(scores).astype(int), valid_min, valid_max)
        post_mean = float(int_scores.mean())
        post_sd   = float(int_scores.std())

        mean_err = target_mean - post_mean
        sd_ratio = target_sd / post_sd if post_sd > 0 else 1.0

        # Check convergence
        if abs(mean_err) < 0.005 and abs(sd_ratio - 1.0) < 0.002:
            break

        # Scale deviations around current post_mean, then shift to target_mean
        scores = (scores - post_mean) * sd_ratio + target_mean

    return np.clip(np.round(scores).astype(int), valid_min, valid_max)


def main():
    np.random.seed(42)
    random.seed(42)

    N = 394

    # Parameters from paper
    means = np.array([29.87, 81.84, 86.10])
    sds   = np.array([ 5.24, 31.49, 12.27])

    # Correlations (from Table 2)
    r_rses_iss   =  0.46
    r_rses_rssis = -0.41
    r_iss_rssis  = -0.49

    # Correlation matrix: order [RSES, RSSIS, ISS]
    corr_matrix = np.array([
        [1.0,          r_rses_rssis, r_rses_iss ],
        [r_rses_rssis, 1.0,          r_iss_rssis],
        [r_rses_iss,   r_iss_rssis,  1.0        ]
    ])

    cov_matrix = np.outer(sds, sds) * corr_matrix

    # Generate data with exact moment matching (Cholesky orthogonalization)
    raw_data     = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=N)
    raw_centered = raw_data - np.mean(raw_data, axis=0)
    current_cov  = np.cov(raw_centered, rowvar=False)
    L_target     = np.linalg.cholesky(cov_matrix)
    L_sample     = np.linalg.cholesky(current_cov)
    data         = raw_centered @ np.linalg.inv(L_sample).T @ L_target.T + means

    # --- Calibrate to bounded integer scores matching paper's M and SD ---
    # Valid ranges: RSES 10-40, RSSIS 36-180, ISS 24-120
    rses_scores  = calibrate_integer_scores(data[:, 0], 29.87,  5.24, 10,  40)
    rssis_scores = calibrate_integer_scores(data[:, 1], 81.84, 31.49, 36, 180)
    iss_scores   = calibrate_integer_scores(data[:, 2], 86.10, 12.27, 24, 120)

    # --- Generate item responses ---
    rses_data  = []   # 10 items, 1–4
    rssis_data = []   # 36 items, 1–5
    iss_data   = []   # 24 items, 1–5

    # Reversed items (1-indexed)
    rses_rev = [3, 5, 8, 9, 10]
    iss_rev  = [2, 4, 7, 9, 12, 15, 18, 20, 22]

    for i in range(N):
        rses_items  = generate_items_for_total(rses_scores[i],  10, 1, 4)
        rssis_items = generate_items_for_total(rssis_scores[i], 36, 1, 5)
        iss_items   = generate_items_for_total(iss_scores[i],   24, 1, 5)

        # Un-reverse: store as raw survey response (high = negative wording)
        for idx in rses_rev:
            rses_items[idx - 1] = 5 - rses_items[idx - 1]
        for idx in iss_rev:
            iss_items[idx - 1]  = 6 - iss_items[idx - 1]

        rses_data.append(rses_items)
        rssis_data.append(rssis_items)
        iss_data.append(iss_items)

    rses_cols = [f"RSES_{i}_REV" if i in rses_rev else f"RSES_{i}" for i in range(1, 11)]
    iss_cols  = [f"ISS_{i}_REV"  if i in iss_rev  else f"ISS_{i}"  for i in range(1, 25)]

    df_rses  = pd.DataFrame(rses_data,  columns=rses_cols)
    df_rssis = pd.DataFrame(rssis_data, columns=[f"RSSIS_{i}" for i in range(1, 37)])
    df_iss   = pd.DataFrame(iss_data,   columns=iss_cols)

    # --- Covariate generation (exact counts from paper, N=394) ---

    # Gender: Male=188 (47.7%), Female=206 (52.3%)
    # Generated correlated with RSSIS and ISS to match the paper's regression:
    #   β(gender→RSSIS) ≈ −.26  (females have lower acculturative stress)
    #   β(gender→ISS)   ≈ +.15  (females have higher intercultural sensitivity)
    #
    # Method: probit-style latent score from z-scored scales.
    # Biserial correction: r_pb = r_latent × φ(z_c)/√(p·q)
    #   φ(Φ⁻¹(0.477)) ≈ 0.399,  √(0.477·0.523) ≈ 0.499  →  factor ≈ 0.799
    # Solving the 2×2 linear system (accounting for r(RSSIS,ISS)=−0.49) gives:
    #   a = −0.306 (weight on z_rssis), b = 0.038 (weight on z_iss)
    #   noise_std = √(1 − 0.1065) ≈ 0.945
    z_rssis = (rssis_scores - rssis_scores.mean()) / rssis_scores.std()
    z_iss   = (iss_scores   - iss_scores.mean())   / iss_scores.std()
    gender_latent = (-0.306 * z_rssis
                     + 0.038 * z_iss
                     + 0.945 * np.random.normal(0, 1, N))
    # Assign exact counts: 188 lowest scores → Male (1), 206 highest → Female (2)
    gender_arr             = np.full(N, 2, dtype=int)           # default: female
    gender_arr[np.argsort(gender_latent)[:188]] = 1            # 188 males

    # Academic Year: 1yr=38, 2yr=159, 3yr=101, 4yr=96  (shuffled randomly)
    year_arr   = np.array([1]*38  + [2]*159 + [3]*101 + [4]*96)
    # TOPIK: Beginner=49, Advanced=345
    topik_arr  = np.array([0]*49  + [1]*345)
    # Economic Status: Low=25, Mid=290, High=79
    eco_arr    = np.array([1]*25  + [2]*290 + [3]*79)

    np.random.shuffle(year_arr)
    np.random.shuffle(topik_arr)
    np.random.shuffle(eco_arr)

    df_covariates = pd.DataFrame({
        "Gender":          gender_arr,
        "Academic_Year":   year_arr,
        "TOPIK_Level":     topik_arr,
        "Economic_Status": eco_arr
    })

    # Combine (no Total columns — compute from items during analysis)
    df_combined = pd.concat([df_rses, df_rssis, df_iss, df_covariates], axis=1)

    # --- Save CSVs ---
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    df_covariates.to_csv(os.path.join(output_dir, "data_covariates_simulated.csv"), index=False)
    df_rses.to_csv(      os.path.join(output_dir, "data_rses_simulated.csv"),       index=False)
    df_rssis.to_csv(     os.path.join(output_dir, "data_rssis_simulated.csv"),      index=False)
    df_iss.to_csv(       os.path.join(output_dir, "data_iss_simulated.csv"),        index=False)
    df_combined.to_csv(  os.path.join(output_dir, "data_combined_simulated.csv"),   index=False)

    # --- Verification report ---
    # Recompute corrected sums (reversals undone) for reporting
    rses_sum  = df_rses.apply(
        lambda row: sum(5 - row[c] if "REV" in c else row[c] for c in rses_cols), axis=1)
    rssis_sum = df_rssis.sum(axis=1)
    iss_sum   = df_iss.apply(
        lambda row: sum(6 - row[c] if "REV" in c else row[c] for c in iss_cols),  axis=1)

    print("Simulation completed.\n")
    print("----- Item-Sum Descriptive Statistics (paper targets in brackets) -----")
    print(f"  RSES : mean={rses_sum.mean():.2f}  [29.87]   SD={rses_sum.std():.2f}  [5.24]")
    print(f"  RSSIS: mean={rssis_sum.mean():.2f} [81.84]  SD={rssis_sum.std():.2f} [31.49]")
    print(f"  ISS  : mean={iss_sum.mean():.2f} [86.10]  SD={iss_sum.std():.2f} [12.27]")

    corr_df = pd.DataFrame({"RSES": rses_sum, "RSSIS": rssis_sum, "ISS": iss_sum})
    print("\n----- Item-Sum Correlation Matrix (paper targets) -----")
    print(f"  RSES-RSSIS: {corr_df['RSES'].corr(corr_df['RSSIS']):.2f}  [-0.41]")
    print(f"  RSES-ISS  : {corr_df['RSES'].corr(corr_df['ISS']):.2f}  [ 0.46]")
    print(f"  RSSIS-ISS : {corr_df['RSSIS'].corr(corr_df['ISS']):.2f}  [-0.49]")

    print("\n----- Covariate Counts (paper targets in brackets) -----")
    g = df_covariates["Gender"].value_counts().sort_index()
    print(f"  Gender  — Male: {g[1]} [188]  Female: {g[2]} [206]")
    y = df_covariates["Academic_Year"].value_counts().sort_index()
    print(f"  Year    — 1: {y[1]} [38]  2: {y[2]} [159]  3: {y[3]} [101]  4: {y[4]} [96]")
    t = df_covariates["TOPIK_Level"].value_counts().sort_index()
    print(f"  TOPIK   — Beg: {t[0]} [49]  Adv: {t[1]} [345]")
    e = df_covariates["Economic_Status"].value_counts().sort_index()
    print(f"  EconSt  — Low: {e[1]} [25]  Mid: {e[2]} [290]  High: {e[3]} [79]")

    # Point-biserial correlations of gender with scales (targets: ≈−.26, ≈+.15)
    g_centered = gender_arr - gender_arr.mean()
    r_g_rssis  = float(np.corrcoef(g_centered, rssis_sum)[0, 1])
    r_g_iss    = float(np.corrcoef(g_centered, iss_sum)[0, 1])
    print(f"\n  r(gender, RSSIS) = {r_g_rssis:.2f}  [target ≈ −0.26]")
    print(f"  r(gender, ISS)   = {r_g_iss:.2f}  [target ≈ +0.15]")


if __name__ == "__main__":
    main()