"""
07_bayes_setup.py
=================
Setup and installation check for the Bayesian analysis pipeline.

This script:
  1. Verifies cmdstanpy is installed and CmdStan is available
  2. Installs CmdStan if needed (with user confirmation)
  3. Compiles all four Stan models
  4. Verifies the data files from 01_generate_data.py exist

Prerequisites:
  pip install cmdstanpy arviz
  python -m cmdstanpy.install_cmdstan   # one-time CmdStan install

Note: This entire Bayesian pipeline requires cmdstanpy and CmdStan.
      Run 01_generate_data.py first to produce the data files.
"""

import sys
import os

out_dir = os.path.dirname(os.path.abspath(__file__))
stan_dir = os.path.join(out_dir, "stan_models")

# ── 1. Check cmdstanpy ────────────────────────────────────────────────────
print("=" * 60)
print("BAYESIAN PIPELINE SETUP")
print("=" * 60)

try:
    import cmdstanpy
    print(f"\n✓ cmdstanpy version: {cmdstanpy.__version__}")
except ImportError:
    print("\n✗ cmdstanpy is NOT installed.")
    print("  Install it with:  pip install cmdstanpy")
    print("  Then install CmdStan:  python -m cmdstanpy.install_cmdstan")
    sys.exit(1)

# ── 2. Check CmdStan ─────────────────────────────────────────────────────
try:
    cmdstan_path = cmdstanpy.cmdstan_path()
    print(f"✓ CmdStan found at: {cmdstan_path}")
except ValueError:
    print("\n✗ CmdStan is NOT installed.")
    resp = input("  Install CmdStan now? (y/n): ").strip().lower()
    if resp == "y":
        print("  Installing CmdStan (this may take several minutes)...")
        cmdstanpy.install_cmdstan()
        print("  ✓ CmdStan installed successfully.")
    else:
        print("  Run: python -m cmdstanpy.install_cmdstan")
        sys.exit(1)

# ── 3. Check arviz ────────────────────────────────────────────────────────
try:
    import arviz
    print(f"✓ arviz version: {arviz.__version__}")
except ImportError:
    print("⚠ arviz is NOT installed (optional, for diagnostics/comparison).")
    print("  Install with: pip install arviz")

# ── 4. Check data files ──────────────────────────────────────────────────
required_data = ["data_items.csv", "data_persons.csv", "data_responses.csv"]
all_present = True
for f in required_data:
    path = os.path.join(out_dir, f)
    if os.path.exists(path):
        print(f"✓ {f} found")
    else:
        print(f"✗ {f} MISSING — run 01_generate_data.py first!")
        all_present = False

if not all_present:
    sys.exit(1)

# ── 5. Compile all Stan models ────────────────────────────────────────────
stan_files = [
    "rasch.stan",
    "latent_regression_rasch.stan",
    "lltm.stan",
    "latent_regression_lltm.stan",
]

print(f"\nCompiling Stan models...")
for sf in stan_files:
    path = os.path.join(stan_dir, sf)
    if not os.path.exists(path):
        print(f"  ✗ {sf} not found in stan_models/")
        continue
    print(f"  Compiling {sf}...", end=" ", flush=True)
    try:
        model = cmdstanpy.CmdStanModel(stan_file=path)
        print(f"✓ ({model.exe_file})")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("SETUP COMPLETE — Ready to run Bayesian analyses (08-12)")
print("=" * 60)
print("""
Pipeline order:
  07_bayes_setup.py                  ← you are here
  08_bayes_rasch.py                  ← Model 1: Rasch
  09_bayes_latent_regression_rasch.py← Model 2: Person explanatory
  10_bayes_lltm.py                   ← Model 3: Item explanatory
  11_bayes_latent_regression_lltm.py ← Model 4: Doubly explanatory
  12_bayes_model_comparison.py       ← LOO-CV comparison of all models
""")
