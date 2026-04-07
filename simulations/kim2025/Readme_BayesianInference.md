# Bayesian Inference Scripts

이 문서는 김연하 외 (2025) 논문 시뮬레이션 데이터에 대해 현재 저장소에 추가된 세 개의 Bayesian 추론 스크립트를 정리한다.

## 파일 개요

1. `simulation_bayesian_regression.py`
   - 목적: 논문의 PROCESS Macro Model 4를 합산점수 기준 Bayesian regression으로 옮긴 버전.
   - 데이터 입력: `rses_simulated.csv`, `iss_simulated.csv`, `rssis_simulated.csv`, `covariates_simulated.csv`.
   - 출력: `mediation_bayesian_regression.stan`, 구조계수 요약, 기본 posterior figure 3개.

2. `simulation_bsem.py`
   - 목적: item-level Bayesian SEM의 PCM 버전.
   - 측정모형: Partial Credit Model 기반 ordinal item model.
   - 구조모형: 잠재 X -> 잠재 M -> 잠재 Y 매개구조 + 중심화된 공변량 회귀.
   - 출력: `mediation_bsem.stan`, PCM 기반 posterior figure 4개.

3. `simulation_ordered_logistic_cfa.py`
   - 목적: item-level 대안모형.
   - 측정모형: ordered logistic CFA with free thresholds and positive factor loadings.
   - 비교기능: `simulation_bsem.py`의 PCM 모형과 컴파일 시간, 샘플링 시간, 주요 구조계수 진단치를 비교.
   - 출력: `mediation_ordered_logit_cfa.stan`, ordered logistic posterior figure 3개, `bayesian_item_model_benchmark.csv`, `fig_bayesian_item_model_benchmark.png`.

## 모델 선택 기준

`simulation_bayesian_regression.py`
- 논문 표 3의 회귀계수 의미와 가장 직접적으로 대응한다.
- 측정오차를 별도로 모델링하지 않으므로 논문 재현용에 적합하다.

`simulation_bsem.py`
- 문항 수준 ordinal response를 직접 사용한다.
- PCM은 문항 단계(step) 구조를 직접 모델링하므로 Likert 범주 이동을 세밀하게 다룰 때 적합하다.

`simulation_ordered_logistic_cfa.py`
- ordered logistic CFA는 구현이 더 단순하고 해석이 직관적이다.
- PCM보다 계산이 빠를 수 있지만, step structure를 분리해서 해석하는 정도는 줄어든다.

## 실행 전 준비

필수 패키지:

```bash
pip install numpy pandas matplotlib arviz cmdstanpy
```

CmdStan이 아직 설치되지 않았다면:

```bash
python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
```

## 실행 예시

합산점수 Bayesian regression:

```bash
python simulation_bayesian_regression.py
```

PCM 기반 item-level BSEM:

```bash
python simulation_bsem.py
```

ordered logistic CFA만 실행:

```bash
python simulation_ordered_logistic_cfa.py --mode ordered-only
```

ordered logistic CFA와 PCM BSEM 비교 실행:

```bash
python simulation_ordered_logistic_cfa.py --mode compare
```

샘플링 횟수 축소 예시:

```bash
python simulation_ordered_logistic_cfa.py --mode compare --iter-warmup 500 --iter-sampling 500
```

## 비교 결과 해석

`bayesian_item_model_benchmark.csv`에는 다음 지표가 저장된다.

- `compile_seconds`: Stan 컴파일 시간
- `sample_seconds`: MCMC 샘플링 시간
- `total_seconds`: 총 시간
- `max_rhat`: 핵심 구조계수 중 가장 큰 R-hat
- `min_ess_bulk`: 핵심 구조계수 중 가장 작은 bulk ESS
- `min_ess_tail`: 핵심 구조계수 중 가장 작은 tail ESS
- `loo_elpd`: leave-one-out 기준 예측적합도
- `waic_elpd`: WAIC 기준 예측적합도

일반적인 해석 기준:

- `max_rhat`가 1.01 부근이면 수렴 상태가 양호한 편이다.
- `min_ess_bulk`, `min_ess_tail`가 클수록 Monte Carlo 오차가 작다.
- 동일한 샘플링 설정에서 `sample_seconds`가 낮고 `max_rhat`가 안정적이면 계산 효율이 더 좋다고 볼 수 있다.
- 같은 데이터에서 `loo_elpd`, `waic_elpd`가 더 큰 모형이 상대적으로 예측적합도가 좋다고 해석할 수 있다.

## 권장 사용 순서

1. 논문 계수 의미를 먼저 확인할 때는 `simulation_bayesian_regression.py`를 사용한다.
2. 문항 수준 잠재구조를 반영하려면 `simulation_bsem.py`를 사용한다.
3. PCM과 ordered logistic CFA의 계산 효율과 안정성을 비교하려면 `simulation_ordered_logistic_cfa.py --mode compare`를 사용한다.

## 주의사항

- item-level 두 모형은 계산량이 크다. 기본 설정으로도 실행 시간이 길 수 있다.
- ordered logistic CFA는 threshold가 item intercept 역할을 흡수하므로, PCM의 step difficulty와 직접 같은 파라미터로 읽으면 안 된다.
- 비교 실험은 동일한 데이터와 동일한 샘플링 설정에서 돌려야 해석이 깔끔하다.