from pathlib import Path

import pandas as pd


def _scale_columns(df, prefix, include_totals):
    columns = [col for col in df.columns if col.startswith(prefix)]
    if not include_totals:
        total_column = f"{prefix[:-1]}_Total"
        columns = [col for col in columns if col != total_column]
    return columns


def load_and_split_combined_data(csv_path="combined_simulated.csv", include_totals=True):
    df = pd.read_csv(csv_path)

    rses_columns = _scale_columns(df, "RSES_", include_totals)
    rssis_columns = _scale_columns(df, "RSSIS_", include_totals)
    iss_columns = _scale_columns(df, "ISS_", include_totals)

    rses = df.loc[:, rses_columns].copy()
    rssis = df.loc[:, rssis_columns].copy()
    iss = df.loc[:, iss_columns].copy()

    used_columns = set(rses.columns) | set(rssis.columns) | set(iss.columns)
    covariates = df.loc[:, ~df.columns.isin(used_columns)].copy()

    return df, rses, rssis, iss, covariates


def main():
    csv_path = Path(__file__).with_name("combined_simulated.csv")
    df, rses, rssis, iss, covariates = load_and_split_combined_data(csv_path, include_totals=False)

    print(f"full data shape: {df.shape}")
    print(f"rses shape: {rses.shape}")
    print(f"rssis shape: {rssis.shape}")
    print(f"iss shape: {iss.shape}")
    print(f"covariates shape: {covariates.shape}")

    print("\nCovariate columns:")
    print(covariates.columns.tolist())


if __name__ == "__main__":
    main()