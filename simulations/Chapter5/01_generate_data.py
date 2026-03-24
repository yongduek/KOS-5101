"""
01_generate_data.py
====================
Simulate data that mirrors the German PISA Mathematical Literacy example
from Chapter 5 (Wilson, De Boeck, & Carstensen, 2008).

Design:
  - 881 students
  - 18 dichotomous items = 3 topic areas × 3 modeling types × 2 items each
  - Person properties: gender (0/1), program (1-4), HiSES (continuous)

The simulation uses a Rasch-like generative model with known parameters
so students can verify their analyses recover the true values.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── 1. Define item design ─────────────────────────────────────────────────
topic_areas = ["Arithmetic", "Geometry", "Algebra"]
modeling_types = ["TechnicalProcessing", "NumericalModeling", "AbstractModeling"]

items = []
item_id = 0
for ta in topic_areas:
    for mt in modeling_types:
        for rep in range(2):  # 2 items per cell
            items.append({
                "item_id": item_id,
                "item_name": f"{ta[:3]}_{mt[:4]}_{rep+1}",
                "topic_area": ta,
                "modeling_type": mt,
            })
            item_id += 1

items_df = pd.DataFrame(items)

# ── 2. True item difficulty parameters (inspired by Table 6 from paper) ───
# These are the "true" β values for each topic_area × modeling_type cell
true_item_effects = {
    ("Arithmetic", "TechnicalProcessing"): -1.16,
    ("Arithmetic", "NumericalModeling"):    0.24,
    ("Arithmetic", "AbstractModeling"):    -0.51,
    ("Algebra",    "TechnicalProcessing"): -0.13,
    ("Algebra",    "NumericalModeling"):    0.69,
    ("Algebra",    "AbstractModeling"):     0.97,
    ("Geometry",   "TechnicalProcessing"): -0.20,
    ("Geometry",   "NumericalModeling"):    0.07,
    ("Geometry",   "AbstractModeling"):     0.58,
}

# Add small noise per item within each cell (two items per cell)
items_df["beta_cell"] = items_df.apply(
    lambda r: true_item_effects[(r["topic_area"], r["modeling_type"])], axis=1
)
noise = np.random.normal(0, 0.08, len(items_df))
items_df["beta_true"] = items_df["beta_cell"] + noise

# ── 3. Person properties ──────────────────────────────────────────────────
N = 881

# Gender: 0 = female, 1 = male (roughly 50/50)
gender = np.random.binomial(1, 0.50, N)

# Program: 1=Hauptschule(22%), 2=Integrated(10%), 3=Realschule(32%), 4=Gymnasium(36%)
program = np.random.choice([1, 2, 3, 4], size=N, p=[0.22, 0.10, 0.32, 0.36])

# HiSES: continuous, standardized (mean~50, sd~10 in original; we use standardized)
hises = np.random.normal(0, 1, N)

# ── 4. True person parameters (latent regression model) ───────────────────
# True effects of person properties (inspired by paper results)
true_program_effects = {1: -0.49, 2: 0.00, 3: 0.80, 4: 1.89}
true_gender_x_program = {1: -0.11, 2: 0.00, 3: -0.22, 4: -0.30}
true_ses_effect = 0.20  # scaled for standardized SES

# Compute the fixed part of person ability
theta_fixed = np.array([true_program_effects[p] for p in program])
theta_gender_prog = np.array([
    true_gender_x_program[p] * (1 - g)  # effect for females (gender=0)
    for p, g in zip(program, gender)
])
theta_ses = true_ses_effect * hises

# Residual random person effect ~ N(0, sigma_e^2)
sigma_e = np.sqrt(0.67)  # residual variance after accounting for predictors
theta_residual = np.random.normal(0, sigma_e, N)

theta_p = theta_fixed + theta_gender_prog + theta_ses + theta_residual

# ── 5. Generate responses ─────────────────────────────────────────────────
beta = items_df["beta_true"].values  # shape (18,)

# η_pi = θ_p - β_i  (logit model)
eta = theta_p[:, None] - beta[None, :]  # shape (N, 18)
prob = 1 / (1 + np.exp(-eta))           # logistic function
responses = (np.random.uniform(size=(N, 18)) < prob).astype(int)

# ── 6. Build output DataFrames ────────────────────────────────────────────
# Person-level data
persons_df = pd.DataFrame({
    "person_id": range(N),
    "gender": gender,
    "program": program,
    "hises": np.round(hises, 4),
    "theta_true": np.round(theta_p, 4),
})

# Response matrix with column names = item names
resp_cols = items_df["item_name"].tolist()
resp_df = pd.DataFrame(responses, columns=resp_cols)
resp_df.insert(0, "person_id", range(N))

# Combined long-format data (useful for GLMM fitting)
long_rows = []
for p in range(N):
    for i in range(18):
        long_rows.append({
            "person_id": p,
            "item_id": i,
            "response": responses[p, i],
        })
long_df = pd.DataFrame(long_rows)

# ── 7. Save files ─────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))

items_df.to_csv(os.path.join(out_dir, "data_items.csv"), index=False)
persons_df.to_csv(os.path.join(out_dir, "data_persons.csv"), index=False)
resp_df.to_csv(os.path.join(out_dir, "data_responses.csv"), index=False)
long_df.to_csv(os.path.join(out_dir, "data_long.csv"), index=False)

print("=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"  Students:  {N}")
print(f"  Items:     {len(items_df)}")
print(f"  Responses: {responses.shape[0]} × {responses.shape[1]}")
print(f"\nFiles saved:")
print(f"  data_items.csv     — item design and true parameters")
print(f"  data_persons.csv   — person properties and true θ")
print(f"  data_responses.csv — binary response matrix (wide)")
print(f"  data_long.csv      — long-format (person × item × response)")
print(f"\nTrue parameter summary:")
print(f"  Person variance (total):    {np.var(theta_p):.3f}")
print(f"  Person variance (residual): {sigma_e**2:.3f}")
print(f"  Item difficulty range:      [{beta.min():.3f}, {beta.max():.3f}]")
print(f"  Mean response rate:         {responses.mean():.3f}")
