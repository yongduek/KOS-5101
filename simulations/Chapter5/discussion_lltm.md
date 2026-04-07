# LLTM (Linear Logistic Test Model) 토론 기록

**출처**: Wilson, De Boeck & Carstensen (2008), Chapter 5 — *Explanatory Item Response Models: A Brief Introduction*

---

## 1. LLTM이란 무엇인가?

### 핵심 아이디어

Rasch 모델에서는 각 문항 *i*마다 별도의 난이도 파라미터 $\beta_i$를 추정한다. 18개 문항이면 18개의 $\beta$를 추정하는 것이다. LLTM은 이 질문을 던진다:

> "왜 어떤 문항은 어렵고 어떤 문항은 쉬운가? 그 난이도를 문항의 속성(item properties)으로 설명할 수 있지 않은가?"

Rasch의 $\beta_i$를 다음으로 대체한다 (논문 Equation 9):

$$\beta'_i = \sum_{k=0}^{K} \beta_k \cdot X_{ik}$$

여기서 $k=0$은 상수항($X_{i0} = 1$ for all $i$)이고, $k=1, \ldots, K$는 실질적인 문항 속성 예측변수다. $\beta_0$를 분리해서 쓰면:

$$\beta'_i = \beta_0 + \sum_{k=1}^{K} \beta_k \cdot X_{ik}$$

이렇게 하면 $k$의 범위가 모호하지 않다. 선형 예측자는 (논문 Equation 8):

$$\eta_{pi} = \theta_p - \sum_{k=0}^{K} \beta_k \cdot X_{ik} = \theta_p - \beta'_i$$

부호 관습: Rasch 모델과 동일하게 $\beta'_i$가 클수록 **어려운** 문항이다 ($\eta_{pi}$가 작아져 정답 확률이 낮아진다).

### 이 논문의 문항 속성 설계

18개 문항은 **3 topic areas × 3 modeling types = 9개 조합**에서 각 2문항씩 만들어졌다.

|             | Technical Processing | Numerical Modeling | Abstract Modeling |
|-------------|---------------------|--------------------|-------------------|
| **Arithmetic** | $\beta_1$ | $\beta_2$ | $\beta_3$ |
| **Algebra**    | $\beta_4$ | $\beta_5$ | $\beta_6$ |
| **Geometry**   | $\beta_7$ | $\beta_8$ | $\beta_9$ |

파라미터 수가 **18개(Rasch)에서 $\beta_1 \sim \beta_9$ + intercept $\beta_0$ = 10개**로 줄어든다.

### $\beta_0$ (intercept)의 역할

모든 문항에 동일하게 적용되는 상수 예측변수 $X_{i0} = 1$을 추가하고, 그 계수 $\beta_0$가 intercept 역할을 한다. 이것은 Rasch 모델에서 $\theta_p$의 평균을 0으로 고정하는 것과 연결된다: $\beta_0$가 문항들의 전반적인 평균 난이도를 흡수한다. $\beta_0$가 없으면 item side에서 원점을 정의할 수 없다.

---

## 2. 각 문항의 예측 난이도 공식

### 9셀 완전 교호작용 코딩 (이 논문의 방식)

각 문항이 **하나의 셀에만 속하도록 9개의 더미 변수를 만드는** 구조다. 이것은 Topic Area와 Modeling Type의 **완전 교호작용(Full Interaction) 모델**과 수학적으로 동일하다: 두 요인의 주효과(main effect)를 분리하지 않고, 9개 조합 각각을 하나의 고유한 범주로 취급한다.

| 문항 소속 | $\beta'_i$ |
|---------|------|
| Arithmetic × Technical Processing | $\beta_0 + \beta_1$ |
| Arithmetic × Numerical Modeling   | $\beta_0 + \beta_2$ |
| Arithmetic × Abstract Modeling    | $\beta_0 + \beta_3$ |
| Algebra × Technical Processing    | $\beta_0 + \beta_4$ |
| Algebra × Numerical Modeling      | $\beta_0 + \beta_5$ |
| Algebra × Abstract Modeling       | $\beta_0 + \beta_6$ |
| Geometry × Technical Processing   | $\beta_0 + \beta_7$ |
| Geometry × Numerical Modeling     | $\beta_0 + \beta_8$ |
| Geometry × Abstract Modeling      | $\beta_0 + \beta_9$ |

X 행렬 ($k=0$부터 $k=9$, 총 10열):

```
item              X₀  X₁  X₂  X₃  X₄  X₅  X₆  X₇  X₈  X₉
Arith×Tech_1       1   1   0   0   0   0   0   0   0   0
Arith×Tech_2       1   1   0   0   0   0   0   0   0   0   ← 동일 행
Arith×Num_1        1   0   1   0   0   0   0   0   0   0
Arith×Num_2        1   0   1   0   0   0   0   0   0   0   ← 동일 행
...
Geom×Abs_1         1   0   0   0   0   0   0   0   0   1
Geom×Abs_2         1   0   0   0   0   0   0   0   0   1   ← 동일 행
```

**같은 셀의 두 문항은 X 행이 완전히 동일하다** → 예측 난이도가 동일하다.

### 대안: 주효과 가산 모델 (더 간결하지만 더 강한 가정)

Topic Area 3개와 Modeling Type 3개를 각각 독립적인 더미 변수로 코딩하면 (각 요인에서 기준 범주 1개 제외):

$$\beta'_i = \beta_0 + \alpha_{\text{topic}(i)} + \gamma_{\text{modeling}(i)}$$

파라미터 수: $\beta_0$ + 2개(Topic 더미) + 2개(Modeling 더미) = **5개**. 이 모델은 Topic과 Modeling Type의 효과가 **독립적으로 가산된다(additivity)**고 가정한다. 교호작용이 없다는 더 강한 제약이다. 논문 p.105는 이 5-파라미터 모델을 시도했을 때 어떤 적합도 지수도 LLTM을 지지하지 않았다고 보고한다.

| 모델 유형 | 파라미터 수 | 교호작용 가정 |
|---------|-----------|------------|
| Rasch (문항 지시자) | 19 | 없음 (각 문항 자유 추정) |
| LLTM — 완전 교호작용 (이 논문) | 10 | Topic × Modeling 교호작용 완전 포함 |
| LLTM — 주효과만 (가산 모델) | 5 | Topic + Modeling 효과가 독립 가산 |

---

## 3. 핵심 가정과 그 이론적 성격

### "문항 파라미터에 대한 구조 모델 (Structural Model on Item Parameters)"

LLTM의 핵심 가정을 정확히 이해하려면 두 가지 층위를 구분해야 한다:

- **응답 데이터 층위**: 각 응시자-문항 쌍 $(p, i)$의 응답 $Y_{pi}$는 여전히 Bernoulli 분포를 따른다. 데이터에 오차가 없다는 뜻이 아니다.
- **문항 파라미터 층위**: LLTM은 $\beta_i$ 자체에 선형 제약(linear constraint)을 가한다. 즉, $\beta_i$들이 자유 파라미터가 아니라 소수의 속성 효과 $\beta_k$의 선형 결합이라고 제약한다.

따라서 LLTM은 본질적으로 **"문항 파라미터 공간에 대한 구조 모델(structural model)"**이다. 응시자 응답의 무작위성은 Rasch와 완전히 동일하게 유지된다.

LLTM이 강한 가정이라고 말하는 것은 **문항 파라미터 수준의 오차 항이 없다**는 의미다. 논문이 직접 밝히는 부분:

> "a strong assumption and it makes the model highly restrictive" (p.104)

### 같은 셀에 난이도가 다른 두 문항이 있다면 X 행렬은 어떻게 변하는가?

세 가지 선택지가 있다:

**Option 1: replication dummy 추가 (파라미터 1개 추가)**

각 셀 안의 두 번째 문항에 $X_{\text{rep}} = 1$을 부여:

```
item              X₀  X₁ ... X₉  X_rep
Arith×Tech_1       1   1 ...  0    0
Arith×Tech_2       1   1 ...  0    1   ← 추가 파라미터 δ
```

$\beta'_{i1} = \beta_0 + \beta_k$, $\beta'_{i2} = \beta_0 + \beta_k + \delta$. 단, $\delta$가 모든 9개 셀에서 동일하다고 가정하는 강한 제약이다.

**Option 2: 셀별 within-cell dummy 9개 추가**

각 셀마다 고유한 $\delta_k$를 허용하면 파라미터가 9개 더 늘어나 총 19개가 된다. 이는 **Rasch 모델과 동일**해진다. LLTM의 이점이 사라진다.

**Option 3: Item random effect 추가 (Relaxed LLTM / LLTM-EA)**

각 문항에 랜덤 오차를 허용:

$$\beta_i = \beta'_i + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2_{\text{item}})$$

Janssen, Shepers & Peres (2004)가 제안한 방식으로, 현대 심리측정학(Psychometrics)에서는 **LLTM-EA (Error Augmented)** 또는 **LLTM with item error**로 불리며 훨씬 더 권장된다. 파라미터 1개($\sigma^2_{\text{item}}$)만 추가하면서 모든 문항이 셀 예측값 주변에서 무작위로 흩어질 수 있다. **가장 현실적인 타협안**.

---

## 4. 동일 난이도 문항 제작의 현실적 불가능성

### 왜 불가능에 가까운가?

문항 난이도($\beta_i$)는 문항 제작자가 직접 설정하는 값이 아니다. 난이도는 문항의 표면적 속성과 응시자 집단의 상호작용에서 **사후적으로 측정되는** 값이다.

같은 "Arithmetic × Technical Processing" 조합이라도 다음 요소들이 달라지면 난이도가 달라진다:

- 구체적인 숫자 선택
- 문장 길이와 어휘 수준
- 중간 계산 단계의 수
- 그림이나 표의 유무
- 문맥 (돈 계산인가, 속도 계산인가)

이것들은 topic area와 modeling type으로는 포착되지 않는 **문항 고유의 분산**이다.

데이터 생성 코드(01_generate_data.py)도 이를 반영한다:

```python
noise = np.random.normal(0, 0.08, len(items_df))
items_df["beta_true"] = items_df["beta_cell"] + noise
```

실제 PISA 같은 표준화 시험에서는 이 노이즈가 $\sigma = 0.08$보다 훨씬 크게 나타난다.

### LLTM이 그럼에도 성립하는 조건

셀 내 분산(within-cell variance)이 셀 간 분산(between-cell variance)에 비해 **상대적으로 작을 때** LLTM은 좋은 근사가 된다. 이 논문에서 Rasch $\beta_i$와 LLTM $\beta'_i$의 상관계수 $r = 0.98$은 매우 높은 수치다:

$$R^2 = r^2 \approx 0.96$$

이는 9개의 셀 효과($\beta_0 \sim \beta_9$)가 18개 문항 난이도 분산의 **96%를 설명**한다는 의미다. 이 연구에서 문항들이 topic × modeling type 조합에 따라 매우 정교하게 설계되었음을 시사한다. 나머지 4%가 문항 고유 노이즈다.

---

## 5. LLTM의 진짜 존재 이유

### "검증 도구"로서의 LLTM은 과장이다

다음 가설: "Algebra × Abstract Modeling은 어렵고 Arithmetic × Technical Processing은 쉽다"를 검증하는 데에는 **Rasch 모델로 충분**하다. Rasch로 $\beta_i$를 추정한 뒤 두 그룹의 평균을 비교하면 된다. LLTM이 굳이 필요하지 않다.

### LLTM의 실질적 강점

| 목적 | Rasch로 충분한가? |
|------|-----------------|
| 기존 문항들의 난이도 비교 | ✔ 충분 |
| 설계 의도 사후 검증 | ✔ 충분 |
| **새 문항의 난이도 사전 예측** | ✗ **LLTM 필요** |
| **표본 작고 문항 많을 때 안정적 추정** | ✗ **LLTM 유리** |
| 이 논문에서의 실제 역할 | 교육적 예시 |

**LLTM의 진정한 강점은 예측(prediction)에 있다:**

Rasch 모델은 이미 시행된 문항의 $\beta_i$만 추정한다. 반면 LLTM이 성립한다면, 아직 시행하지 않은 새 문항이라도 그것이 어떤 topic × modeling type 조합인지만 알면 난이도를 미리 예측할 수 있다:

$$\text{새 문항} \in \text{(Algebra} \times \text{Abstract Modeling)} \implies \hat{\beta}'_i = \hat{\beta}_0 + \hat{\beta}_6 \approx 0.97 \quad \text{(시행 전에 알 수 있음)}$$

이것은 **문항 은행(item banking)과 자동화된 검사 조립(automated test assembly)** 에서 핵심적으로 사용되는 기능이다.

---

## 6. 적합도 비교 (논문 Table 5)

| 모델 | Deviance | AIC | BIC | 파라미터 수 |
|------|---------|-----|-----|----------|
| Rasch | 18680.3 | 18718.3 | 18809.1 | 19 |
| LLTM  | 18721.5 | 18741.5 | **18789.3** | 10 |

### 해석 시 주의사항

LLTM의 목적은 Rasch보다 "더 잘 맞는 모델"을 찾는 것이 **아니다**. Rasch 모델은 문항별 자유 파라미터를 가지므로 항상 데이터에 더 잘 맞는다(Deviance 항상 낮음). LLTM의 진짜 질문은:

> "약간의 적합도를 희생하더라도, 문항 난이도를 소수의 속성 효과로 얼마나 잘 **설명(explain)**할 수 있는가?"

이 관점에서 적합도 지수를 재해석하면:

- **Deviance, AIC**: LLTM이 열등 → 10개 파라미터로 19개 파라미터의 모델만큼 응답 데이터에 맞추지는 못함 (당연한 결과)
- **BIC**: LLTM이 우세 → BIC 패널티 계수 $\ln(N) \approx \ln(881) \approx 6.78$이 AIC의 2보다 크기 때문에, 9개 파라미터 절감이 적합도 손실을 충분히 상쇄
- **$R^2 = 0.96$**: LLTM의 실질적 설명력 지표 — 9개 설계 요인이 18개 문항 난이도 분산의 96%를 설명함. 이것이 LLTM이 이 데이터에서 의미 있는 모델임을 지지하는 핵심 근거다.

---

## 7. 결론

LLTM은 "동일 난이도 문항을 만들 수 있는 전문가"를 요구하는 모델이 아니다.

- **이론적 성격**: 문항 파라미터 공간에 선형 제약을 가하는 구조 모델. 응답 데이터의 무작위성은 Rasch와 동일.
- **근사 모델**로서: 셀 내 분산 $\ll$ 셀 간 분산이면 성립 ($R^2 = 0.96$이 이를 뒷받침)
- **예측 도구**로서: 새 문항의 난이도 사전 예측에 핵심적 역할
- **효율적 추정**으로서: 문항 많고 표본 작을 때 안정적
- **현대적 확장**: LLTM-EA(Error Augmented)로 item random effect를 추가하면 현실적 타협 가능
- **이 논문에서**: 네 가지 모델 유형(Table 3)을 모두 예시하는 교육적 목적

같은 셀 안에서 난이도가 많이 흩어져 있다면 LLTM의 fit이 나빠지고 $R^2$가 낮아지며, 그것 자체가 "설계 의도대로 문항이 작동하지 않는다"는 유용한 진단 정보가 된다.
