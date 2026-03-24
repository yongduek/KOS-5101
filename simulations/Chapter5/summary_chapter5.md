# Chapter 5: Explanatory Item Response Models — A Brief Summary

**Authors:** Mark Wilson, Paul De Boeck, and Claus H. Carstensen

## Overview

This chapter introduces **Explanatory Item Response Models (IRMs)** as special cases of **Generalized Linear Mixed Models (GLMMs)**. It contrasts the traditional *measurement* approach (where items and persons each get their own parameters) with an *explanatory* approach (where item and person properties are used to explain those parameters). The framework is illustrated using data from the **German PISA Mathematical Literacy Test** (881 students, 18 dichotomous items).

## Data: German Mathematical Literacy Example

- **881 students** from the 2003 German PISA sample (15-year-olds).
- **18 dichotomous items** constructed by crossing **3 topic areas** (Arithmetic, Algebra, Geometry) × **3 modeling types** (Technical Processing, Numerical Modeling, Abstract Modeling).
- **Person properties:** Gender (0/1), Program (school track: 1–4), HiSES (socio-economic status).

## The GLMM Framework

All item response models share three components:

1. **Linear component** — a linear predictor η combining fixed and random effects.
2. **Link function** — connects the expected response π to η (logit link → logistic models; probit link → normal-ogive models).
3. **Random component** — the distribution of the observed response Y (Bernoulli for binary data).

## Four Models Presented

| Model | Person Side | Item Side | Type |
|-------|------------|-----------|------|
| **Rasch Model** | Person indicator (θ_p random) | Item indicators (β_i fixed) | Doubly descriptive |
| **Latent Regression Rasch** | Person properties (gender, program, SES) + θ_p | Item indicators | Person explanatory |
| **LLTM** | Person indicator (θ_p random) | Item properties (topic area × modeling type) | Item explanatory |
| **Latent Regression LLTM** | Person properties + θ_p | Item properties | Doubly explanatory |

### 1. Rasch Model (Doubly Descriptive)

- η_pi = θ_p − β_i, with θ_p ~ N(0, σ²)
- Estimated person variance: **1.561**; item parameters range: −1.148 to +1.277
- Reliability: **.79**

### 2. Latent Regression Rasch Model (Person Explanatory)

- η_pi = Σ ϑ_j Z_pj + θ_p − β_i
- Adds gender, program, SES (and gender × program interactions) as person predictors.
- Program effects are highly significant; higher program → higher ability.
- Gender × program interaction: females underperform at top two program levels.
- Person variance explained by predictors: **61%**; residual variance: **0.668**.

### 3. LLTM (Item Explanatory)

- η_pi = θ_p − Σ β_k X_ik
- Replaces individual item parameters with effects of topic area × modeling type.
- Correlation between LLTM predictions and Rasch estimates: **r = .98**.
- BIC favors LLTM over Rasch (more parsimonious).

### 4. Latent Regression LLTM (Doubly Explanatory)

- Combines both person and item explanations.
- Best deviance and AIC among all four models.
- Residual person variance: **0.663**.

## Model Comparison (Fit Indices)

| Model | Deviance | AIC | BIC |
|-------|----------|-----|-----|
| Rasch | 18680.3 | 18718.3 | 18809.1 |
| Latent Regression Rasch | 17570.1 | 17624.1 | 17753.2 |
| LLTM | 18721.5 | 18741.5 | 18789.3 |
| Latent Regression LLTM | 17608.2 | 17644.2 | 17730.3 |

## Key Takeaways

1. Item response models are special cases of GLMMs — this connects psychometrics to mainstream statistics.
2. Moving from *descriptive* to *explanatory* models lets us understand **why** persons and items differ, not just **that** they differ.
3. Person properties (gender, program, SES) explained 61% of person variance in mathematical literacy.
4. Item properties (topic area × modeling type) predicted Rasch item parameters almost perfectly (r = .98).
5. The GLMM perspective makes it straightforward to combine person-side and item-side explanations into a single model.
