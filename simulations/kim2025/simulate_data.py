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
            
    # Optional: add a bit of noise (swap +1 and -1) to increase variance and "natural" look
    # without changing the total sum, but bounded.
    for _ in range(num_items * 2):
        i, j = random.sample(range(num_items), 2)
        if items[i] < max_val and items[j] > min_val:
            items[i] += 1
            items[j] -= 1
            
    return items

def main():
    np.random.seed(42)
    random.seed(42)
    
    N = 394
    
    # Parameters from paper
    means = np.array([29.87, 81.84, 86.10])
    sds = np.array([5.24, 31.49, 12.27])
    
    # Correlations
    r_rses_iss = 0.46
    r_rses_rssis = -0.41
    r_iss_rssis = -0.49
    
    # Covariance Matrix: cov(x,y) = r * sd(x) * sd(y)
    corr_matrix = np.array([
        [1.0,           r_rses_rssis,  r_rses_iss],
        [r_rses_rssis,  1.0,           r_iss_rssis],
        [r_rses_iss,    r_iss_rssis,   1.0]
    ])
    
    cov_matrix = np.outer(sds, sds) * corr_matrix
    
    # Generate raw data and force exact moment matching
    raw_data = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=N)
    raw_centered = raw_data - np.mean(raw_data, axis=0)
    current_cov = np.cov(raw_centered, rowvar=False)
    L_target = np.linalg.cholesky(cov_matrix)
    L_sample = np.linalg.cholesky(current_cov)
    data = raw_centered @ np.linalg.inv(L_sample).T @ L_target.T + means
    
    # Generate item responses
    rses_data = []    # 10 items, 1-4
    rssis_data = []   # 36 items, 1-5
    iss_data = []     # 24 items, 1-5
    
    # Negative (Reversed) Item Definition
    rses_rev = [3, 5, 8, 9, 10]
    iss_rev = [2, 4, 7, 9, 12, 15, 18, 20, 22]
    
    for i in range(N):
        rses_score = data[i, 0]
        rssis_score = data[i, 1]
        iss_score = data[i, 2]
        
        rses_items = generate_items_for_total(rses_score, 10, 1, 4)
        rssis_items = generate_items_for_total(rssis_score, 36, 1, 5)
        iss_items = generate_items_for_total(iss_score, 24, 1, 5)
        
        # Un-reverse the physical generated scores so they represent raw survey input!
        for idx in rses_rev:
            rses_items[idx-1] = 5 - rses_items[idx-1]
        for idx in iss_rev:
            iss_items[idx-1] = 6 - iss_items[idx-1]
        
        rses_data.append(rses_items)
        rssis_data.append(rssis_items)
        iss_data.append(iss_items)
        
    rses_cols = [f"RSES_{i}_REV" if i in rses_rev else f"RSES_{i}" for i in range(1, 11)]
    iss_cols = [f"ISS_{i}_REV" if i in iss_rev else f"ISS_{i}" for i in range(1, 25)]
    
    df_rses = pd.DataFrame(rses_data, columns=rses_cols)
    df_rssis = pd.DataFrame(rssis_data, columns=[f"RSSIS_{i}" for i in range(1, 37)])
    df_iss = pd.DataFrame(iss_data, columns=iss_cols)
    
    # Combine dataset
    df_combined = pd.concat([df_rses, df_rssis, df_iss], axis=1)
    
    # Use exact continuous scores for totals to maintain exact equality with paper
    df_combined["RSES_Total"] = data[:, 0]
    df_combined["RSSIS_Total"] = data[:, 1]
    df_combined["ISS_Total"] = data[:, 2]    

    # --- ADD COVARIATES SIMULATION (Empirical Ratios from Paper) ---
    # Gender (1:Male, 2:Female) | Prob: 0.477, 0.523
    cov_gender = np.random.choice([1, 2], size=N, p=[0.477, 0.523])
    # Academic Year (1-4) | Prob: 0.096, 0.404, 0.256, 0.244
    cov_year = np.random.choice([1, 2, 3, 4], size=N, p=[0.096, 0.404, 0.256, 0.244])
    # TOPIK (0:Beg, 1:Adv) | Prob: 0.124, 0.876
    cov_topik = np.random.choice([0, 1], size=N, p=[0.124, 0.876])
    # Economic Status (1:Low, 2:Med, 3:High) | Prob: 0.063, 0.736, 0.201
    cov_eco = np.random.choice([1, 2, 3], size=N, p=[0.063, 0.736, 0.201])
    
    df_covariates = pd.DataFrame({
        "Gender": cov_gender,
        "Academic_Year": cov_year,
        "TOPIK_Level": cov_topik,
        "Economic_Status": cov_eco
    })
    
    # Merge covariates into the combined dataframe
    df_combined = pd.concat([df_combined, df_covariates], axis=1)

    # Save CSVs
    output_dir = "."  # Fix hardcoded path to current directory
    os.makedirs(output_dir, exist_ok=True)
    df_covariates.to_csv(os.path.join(output_dir, "covariates_simulated.csv"), index=False)
    df_rses.to_csv(os.path.join(output_dir, "rses_simulated.csv"), index=False)
    df_rssis.to_csv(os.path.join(output_dir, "rssis_simulated.csv"), index=False)
    df_iss.to_csv(os.path.join(output_dir, "iss_simulated.csv"), index=False)
    df_combined.to_csv(os.path.join(output_dir, "combined_simulated.csv"), index=False)
    
    print("Simulation completed.")
    print("----- Descriptive Statistics -----")
    print(df_combined[["RSES_Total", "RSSIS_Total", "ISS_Total"]].describe())
    print("\n----- Correlation Matrix -----")
    print(df_combined[["RSES_Total", "RSSIS_Total", "ISS_Total"]].corr())

if __name__ == "__main__":
    main()
