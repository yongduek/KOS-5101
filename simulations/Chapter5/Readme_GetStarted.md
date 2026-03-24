# Getting Started: Explanatory Item Response Models

**Chapter 5 — Wilson, De Boeck & Carstensen (2008)**
*A Brief Introduction to Explanatory IRT Models as Generalized Linear Mixed Models*

This folder contains a complete simulation-based learning pipeline for the four Explanatory IRT models presented in Chapter 5. You will generate data, explore it visually, fit models using two paradigms (frequentist MML and Bayesian MCMC), and compare results.

---

## Prerequisites

**Python 3.10+** with the following packages:

```bash
pip install numpy pandas matplotlib seaborn
```

For Bayesian inference (Phase 4), you also need:

```bash
pip install cmdstanpy arviz
python -m cmdstanpy.install_cmdstan   # one-time setup, installs the C++ toolchain
```

Open notebooks with Jupyter:

```bash
pip install jupyter
jupyter notebook
```

---

## Roadmap — Four Phases

Work through the phases in order. Each phase builds on the previous one.

### Phase 1: Data Appreciation

**Goal:** Understand the simulated dataset before fitting any models.

First, generate the data (run once):

```bash
python 01_generate_data.py
```

This creates four CSV files: `data_items.csv` (18 items), `data_persons.csv` (881 students), `data_responses.csv` (response matrix), and `data_long.csv` (long format).

Then open the exploration notebooks in order:

| Notebook | What You'll See |
|----------|----------------|
| `NB1_Item_Data.ipynb` | The 3×3 item design (topic area × modeling type), true difficulty parameters, heatmaps and box plots of item properties |
| `NB2_Person_Data.ipynb` | Student demographics (gender, school program, SES), ability distributions, violin plots by group, SES-ability scatter |
| `NB3_Response_Data.ipynb` | The raw 881×18 binary response matrix, p-values, sum score distributions, item-total correlations, inter-item correlation heatmap |

**Key questions to answer as you explore:**
- Which topic area has the easiest items? Which modeling type is hardest?
- How does true ability differ across the four school programs?
- What is the overall response rate? Are there ceiling/floor effects?

### Phase 2: Theory — The Four Models

**Goal:** Understand the mathematical structure of Explanatory IRT models.

Open `NB4_Theory.ipynb`. This notebook covers:

1. **The GLMM Framework** — How IRT models are special cases of Generalized Linear Mixed Models (Bernoulli distribution, logit link, random person effects)
2. **Model 1: Rasch** — $\eta_{pi} = \theta_p - \beta_i$ (doubly descriptive, separate parameter per item)
3. **Model 2: Latent Regression Rasch** — $\eta_{pi} = \sum_j \vartheta_j Z_{pj} + \theta_p - \beta_i$ (person explanatory, uses gender/program/SES to predict ability)
4. **Model 3: LLTM** — $\eta_{pi} = \theta_p - \sum_k \beta_k X_{ik}$ (item explanatory, constrains item difficulty by design properties)
5. **Model 4: Latent Regression LLTM** — $\eta_{pi} = \sum_j \vartheta_j Z_{pj} + \theta_p - \sum_k \beta_k X_{ik}$ (doubly explanatory, both sides constrained)

The notebook includes ICC curves, variance decomposition diagrams, and the model comparison table from the chapter (Table 5).

Also read `summary_chapter5.md` for a concise text overview of the chapter.

### Phase 3: MML Estimation (Frequentist)

**Goal:** Fit all four models using Marginal Maximum Likelihood with Gauss-Hermite quadrature.

Work through two notebooks:

| Notebook | Scripts Covered | What You'll Learn |
|----------|----------------|-------------------|
| `NB5_MML_DataGeneration.ipynb` | `01_generate_data.py`, `02_descriptive_analysis.py` | How the simulation works step by step, descriptive statistics |
| `NB6_MML_ModelFitting.ipynb` | `03_rasch_model.py` through `06_latent_regression_lltm.py` | MML estimation from scratch with numpy, all four models, Wright map, ICC curves, model comparison |

**What the code does under the hood:**

- **Gauss-Hermite quadrature** (21 nodes) integrates out the random person effect analytically
- **Gradient descent** with adaptive step size optimizes the marginal log-likelihood (no scipy needed)
- **EAP (Expected A Posteriori)** estimates compute person abilities from the posterior
- **Fit indices** (Deviance, AIC, BIC) enable model comparison

**Key outputs:**

| File | Contents |
|------|----------|
| `results_rasch.csv` | Estimated vs. true item difficulties |
| `results_persons_rasch.csv` | EAP ability estimates for all 881 students |
| `results_person_effects.csv` | Person predictor effects (gender, program, SES) |
| `results_lltm_effects.csv` | Item property effects (topic × modeling) |
| `results_model_comparison.csv` | Deviance, AIC, BIC for all 4 models |
| `fig_wright_map.png` | Wright map (persons vs. items on logit scale) |
| `fig_rasch_vs_lltm.png` | How well LLTM predictions match Rasch estimates |
| `fig_model_comparison.png` | Bar charts of fit indices |
| `fig_icc_curves.png` | Item Characteristic Curves by topic area |

### Phase 4: Bayesian Estimation (Stan/MCMC)

**Goal:** Fit the same four models with Bayesian inference and compare with MML.

First, compile the Stan models (run once):

```bash
python 07_bayes_setup.py
```

Then work through two notebooks:

| Notebook | Scripts Covered | What You'll Learn |
|----------|----------------|-------------------|
| `NB7_Bayes_ModelFitting.ipynb` | `08_bayes_rasch.py` through `11_bayes_latent_regression_lltm.py` | Full posterior distributions, credible intervals, trace plots, variance decomposition with uncertainty |
| `NB8_Bayes_ModelComparison.ipynb` | `12_bayes_model_comparison.py` | LOO-CV (PSIS-LOO), manual WAIC computation, True vs MML vs Bayesian parameter recovery |

**Stan model files** are in `stan_models/`:

| File | Model |
|------|-------|
| `rasch.stan` | $\eta = \theta_p - \beta_i$ |
| `latent_regression_rasch.stan` | $\eta = \sum \vartheta_j Z_{pj} + \theta_p - \beta_i$ |
| `lltm.stan` | $\eta = \theta_p - \sum \beta_k X_{ik}$ |
| `latent_regression_lltm.stan` | $\eta = \sum \vartheta_j Z_{pj} + \theta_p - \sum \beta_k X_{ik}$ |

All models use weakly informative priors: $\text{Normal}(0, 25)$ for fixed effects and $\text{Half-Cauchy}(0, 2.5)$ for scale parameters.

**Important note on visualizations:** All plots in these notebooks use matplotlib and seaborn directly. When arviz is available for LOO computation, the WAIC manual computation is still shown so you can see exactly how the log-likelihood draws are processed into information criteria.

**Key differences from MML (Phase 3):**

| Aspect | MML (Phase 3) | Bayesian (Phase 4) |
|--------|--------------|-------------------|
| Estimation | Marginal ML + quadrature | Full MCMC (HMC/NUTS) |
| Output | Point estimates + SEs | Full posterior distributions |
| Item parameters | Fixed point estimates | Posteriors with credible intervals |
| Person parameters | EAP (post-hoc) | Jointly estimated |
| Model comparison | Deviance, AIC, BIC | LOO-CV (PSIS-LOO), WAIC |
| Uncertainty | Asymptotic SEs | Exact finite-sample posteriors |
| Computation | Fast (~seconds) | Slower (~minutes per model) |

---

## Quick Reference: All Files

### Notebooks (run in order)

| # | Notebook | Phase | Purpose |
|---|----------|-------|---------|
| 1 | `NB1_Item_Data.ipynb` | 1 | Explore item design and difficulty |
| 2 | `NB2_Person_Data.ipynb` | 1 | Explore student demographics and ability |
| 3 | `NB3_Response_Data.ipynb` | 1 | Explore the response matrix |
| 4 | `NB4_Theory.ipynb` | 2 | Mathematical foundations of the four models |
| 5 | `NB5_MML_DataGeneration.ipynb` | 3 | Data simulation walkthrough |
| 6 | `NB6_MML_ModelFitting.ipynb` | 3 | MML estimation of all four models |
| 7 | `NB7_Bayes_ModelFitting.ipynb` | 4 | Bayesian estimation of all four models |
| 8 | `NB8_Bayes_ModelComparison.ipynb` | 4 | Model comparison (LOO, WAIC, parameter recovery) |

### Python Scripts (standalone versions)

| Script | Description |
|--------|-------------|
| `01_generate_data.py` | Simulate the dataset |
| `02_descriptive_analysis.py` | Exploratory analysis |
| `03_rasch_model.py` | MML Rasch model |
| `04_latent_regression_rasch.py` | MML person-explanatory model |
| `05_lltm.py` | MML item-explanatory model |
| `06_latent_regression_lltm.py` | MML doubly-explanatory model |
| `07_bayes_setup.py` | Compile Stan models |
| `08_bayes_rasch.py` | Bayesian Rasch model |
| `09_bayes_latent_regression_rasch.py` | Bayesian person-explanatory model |
| `10_bayes_lltm.py` | Bayesian item-explanatory model |
| `11_bayes_latent_regression_lltm.py` | Bayesian doubly-explanatory model |
| `12_bayes_model_comparison.py` | Bayesian model comparison |

### Documentation

| File | Description |
|------|-------------|
| `Readme_GetStarted.md` | This file — start here |
| `README_bayesian.md` | Details on the Bayesian pipeline |
| `summary_chapter5.md` | Concise summary of the chapter |

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'cmdstanpy'"** — Install it with `pip install cmdstanpy`, then run `python -m cmdstanpy.install_cmdstan`.

**"CmdStan not found"** — Run `python -m cmdstanpy.install_cmdstan` (requires a C++ compiler).

**MML scripts run slowly** — The custom gradient descent optimizer is intentionally simple for pedagogical reasons. Each model takes 1-3 minutes. Iteration progress is printed so you can monitor convergence.

**Figures don't display in notebooks** — Make sure you have `%matplotlib inline` at the top of the notebook.

**CSV files not found** — Run `01_generate_data.py` (or NB5) first. The MML notebooks also save result CSVs needed by the Bayesian notebooks.
