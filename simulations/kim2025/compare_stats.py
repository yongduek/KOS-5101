import pandas as pd
import numpy as np
from scipy import stats

def fisher_z(r):
    # Clip r to prevent division by zero or log of negative numbers if |r| == 1
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlation(r_sample, r_pop, n):
    z_sample = fisher_z(r_sample)
    z_pop = fisher_z(r_pop)
    se = 1 / np.sqrt(n - 3)
    z_stat = (z_sample - z_pop) / se
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    return z_stat, p_val

def main():
    print("Loading data...")
    df = pd.read_csv("combined_simulated.csv")
    
    # Paper statistics
    paper_means = {
        "RSES_Total": 29.87,
        "RSSIS_Total": 81.84,
        "ISS_Total": 86.10
    }
    
    paper_corrs = {
        ("RSES_Total", "RSSIS_Total"): -0.41,
        ("RSES_Total", "ISS_Total"): 0.46,
        ("ISS_Total", "RSSIS_Total"): -0.49
    }
    
    n = len(df)
    print(f"Sample size: {n}\n")
    
    print("=== Mean Comparison (One-Sample T-Test) ===")
    print(f"{'Variable':<15} | {'Sim Mean':<10} | {'Paper Mean':<10} | {'T-stat':<10} | {'P-value':<10}")
    print("-" * 65)
    for var, pop_mean in paper_means.items():
        sample_mean = df[var].mean()
        t_stat, p_value = stats.ttest_1samp(df[var].dropna(), pop_mean)
        print(f"{var:<15} | {sample_mean:<10.3f} | {pop_mean:<10.3f} | {t_stat:<10.3f} | {p_value:<10.3f}")
        
    print("\n=== Correlation Comparison (Fisher Z-Test) ===")
    print(f"{'Variables':<30} | {'Sim Corr':<10} | {'Paper Corr':<10} | {'Z-stat':<10} | {'P-value':<10}")
    print("-" * 80)
    for (var1, var2), pop_corr in paper_corrs.items():
        sample_corr = df[var1].corr(df[var2])
        z_stat, p_value = compare_correlation(sample_corr, pop_corr, n)
        print(f"{var1+', '+var2:<30} | {sample_corr:<10.3f} | {pop_corr:<10.3f} | {z_stat:<10.3f} | {p_value:<10.3f}")
        
    print("\nConclusion: If P-value > 0.05, there is no significant difference between the simulated data and the paper's statistics, meaning they are statistically 'equal'.")

if __name__ == '__main__':
    main()
