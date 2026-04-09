# kim2025

Simulation of Kim 2025 paper.

## Data generation
- Run `sim_generate_data.py` to create CSV data files. 
- The statistics will be almost the same as those reported in the paper.
- The items with revered question is denoted by column names with '_REV' suffix.

## Bayesian validation summary

Three Bayesian variants were implemented and executed in the global Python 3.12 environment.

### 1. Summed-score Bayesian regression

Script: `simulation_bayesian_regression.py`

Validation run settings:
- chains = 2
- warmup = 300
- sampling = 300

Observed reconstructed totals from the simulated item data:
- RSES mean = 29.787
- ISS mean = 86.094
- RSSIS mean = 82.779
- Corr(RSES, ISS) = 0.458
- Corr(RSES, RSSIS) = -0.401
- Corr(ISS, RSSIS) = -0.502

Posterior summary from the validation run:
- a = 1.156
- b = -0.727
- cp = -0.795
- indirect = -0.841
- total = -1.636

Generated outputs:
- `fig_bayes_regression_trace.png`
- `fig_bayes_regression_indirect.png`
- `fig_bayes_regression_forest.png`

### 2. Item-level PCM BSEM

Script: `simulation_bsem.py`

Validation run settings:
- chains = 2
- warmup = 10
- sampling = 10

Posterior summary from the validation run:
- a = 0.204
- b = -0.878
- cp = -0.288
- indirect = -0.205
- total = -0.493

Notes from the validation run:
- The script completed and wrote figures successfully.
- CmdStan reported non-fatal warnings involving zero scale values during some proposals.
- R-hat values were not reliable because this run used very small iteration counts for smoke testing.

Generated outputs:
- `fig_bsem_pcm_trace.png`
- `fig_bsem_pcm_indirect.png`
- `fig_bsem_pcm_forest.png`
- `fig_bsem_pcm_covariates.png`

### 3. Ordered logistic CFA alternative with benchmark comparison

Script: `simulation_ordered_logistic_cfa.py`

Validation run settings:
- chains = 2
- warmup = 10
- sampling = 10
- mode = compare

Posterior summary from the ordered logistic validation run:
- a = 0.503
- b = -0.730
- cp = -0.455
- indirect = -0.362
- total = -0.817

Benchmark comparison against PCM BSEM from `bayesian_item_model_benchmark.csv`:

| Model | Compile sec | Sample sec | Max R-hat | LOO elpd | WAIC elpd |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordered_logit_cfa | 17.39 | 78.36 | 1.811 | -39100.71 | -39465.15 |
| pcm_bsem | 16.00 | 39.99 | 1.698 | -49076.87 | -84242.85 |

Notes from the validation run:
- The script completed and wrote the benchmark CSV and figure successfully.
- Ordered logistic sampling produced non-fatal cut-point ordering warnings during some proposals.
- ArviZ reported LOO and WAIC reliability warnings because the validation run used too few posterior draws.
- These benchmark numbers are useful as execution proof, not as final model-selection evidence.

Generated outputs:
- `fig_ordered_logit_trace.png`
- `fig_ordered_logit_indirect.png`
- `fig_ordered_logit_forest.png`
- `bayesian_item_model_benchmark.csv`
- `fig_bayesian_item_model_benchmark.png`

## Interpretation

The three scripts now run end-to-end and generate their expected artifacts. However, only the summed-score Bayesian regression was run with enough draws to be mildly interpretable. The two item-level models were run in reduced smoke-test mode, so convergence diagnostics and predictive-fit criteria should not be treated as final.

## TODO

1. Re-run the ordered logistic CFA and PCM BSEM with substantially more warmup and sampling draws so that R-hat, ESS, WAIC, and LOO become interpretable.
2. Tighten the item-level model parameterization with stronger priors or safer threshold/scale constraints to reduce the non-fatal Stan warnings seen in the smoke-test runs.
