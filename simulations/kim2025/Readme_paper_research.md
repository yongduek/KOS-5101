# 📄 통계 방법론 및 논문(김연하 외, 2025) 연구 질의응답 리포트

> **💡 학습자 안내 (Student Note)**
> 본 문서는 논문의 코어 통계 방법론(Mediation, Covariates, Dummy Coding, BSEM 등)에 대한 핵심 설명과 함께, 이해를 돕기 위한 **심화 논의 및 직관적인 개념 해석(Q&A)**이 각 세션에 추가로 통합되어 있습니다.

---

## 1. SPSS PROCESS Macro에 대해서 설명해줘
SPSS PROCESS Macro는 Andrew F. Hayes 박사가 개발한 통계 분석 플러그인 모듈로, 주로 **매개효과(Mediation Effect)**, **조절효과(Moderation Effect)**, 그리고 두 가지가 결합된 **조절된 매개효과(Moderated Mediation)**를 분석하는 데 널리 사용됩니다.
* **템플릿 제공:** 90여 개 이상의 복잡한 모형(Model 번호) 설정이 이미 수식화되어 내장되어 있어 복잡한 식 릴레이 없이 쉽게 모형화(Modeling)가 가능합니다.
* **부트스트래핑(Bootstrapping):** 간접효과(Indirect Effect) 등 검증이 까다로운 간섭 모형에 대해 비모수 부트스트래핑 기반 신뢰구간(Confidence Interval, CI) 분석을 자동으로 처리해 줍니다.

## 2. 3_김연하 pdf 에서 사용한 방법은 어떤 통계 모델인지 설명하라
해당 논문에서 사용된 것은 SPSS PROCESS Macro의 **Model 4 (단순 매개 모형, Simple Mediation Model)**입니다.
이는 1개의 독립변수(Independent Variable, 자아존중감)가 1개의 종속변수(Dependent Variable, 문화적응스트레스)로 가는 경로 사이에 1개의 매개변수(Mediator, 상호문화감수성)가 위치하여 어떤 중간 메커니즘을 통해 매개하는지 그 크기와 방향은 어떤지를 분석하는 통계 모형(Statistical Model)입니다.

## 3. 이 논문에서 세 가지 서베이의 총합(total score)을 사용한 이유를 설명하라.
이 논문은 고전검사이론(Classical Test Theory, CTT) 패러다임 기반의 **일반최소제곱법(Ordinary Least Squares, OLS) 다중회귀분석(Multiple Regression Analysis)** 패키지인 PROCESS Macro를 분석 도구로 선정했기 때문입니다.
회귀 모형(Regression Model)에는 다문항으로 이루어진 측정변수(Observed Variable)들의 차원을 한 번에 투입할 수 없으므로, 연구자들은 어쩔 수 없이 각 설문 문항들의 합계점수(Total Score)나 문항 평균(Mean)을 **단 하나의 대푯값인 단일 연속형 수치(Continuous Scalar)**로 뭉뚱그려(Summary) 방정식을 단순하게 만들어 입력할 수밖에 없었습니다.

## 4. Mediation analysis 개념과 연결하여 이 논문을 설명하라
매개 분석(Mediation Analysis)은 독립변수(X)가 무조건적으로 종속변수(Y)를 만들어내는 블랙박스가 아니라, **도대체 "왜", "어떤 경로를 통해(Mechanism)"** Y로 이어지는지를 파헤칩니다. 이 논문 모델에 대입하면 다음과 같습니다.
* **총효과(Total Effect):** 자아존중감(X)이 높은 학생일수록 평균적으로 문화적응스트레스(Y)가 더 낮게 관측됩니다.
* **메커니즘 발굴(Mediation):** 자존감이 높은 학생들은 낯선 문화에 대한 타인 수용력인 상호문화감수성(M)이 높아지게 되고(Path $a$), 그렇게 만들어진 훌륭한 타문화 감수성이 결과적으로 문화적응에서의 혼란과 스트레스(Y)를 낮춰주는 방어막 역할(Path $b$)을 합니다.

> **💬 심화 논의: M이 Y에 '부가적으로 추가'된다는 의미?**
> 매개변수(Mediator) M은 결국 **'설명력 나누기'**의 핵심입니다. 종속변수(Y)를 예측하는 모형에 X 하나만 넣었을 때(총효과)와, Y 모형에 X와 함께 M을 '추가'하였을 때를 비교합니다. M이 강한 예측력을 발휘하면서 애초에 X 혼자 뽐내던 Y에 대한 영향력인 직접효과(Direct Effect)가 눈에 띄게 줄어든다면, *"X가 Y에 주던 영향 중 상당 부분이 사실은 M이 중간에서 전달하던 역할(징검다리)이었구나!"* 라고 판단합니다.

## 5. Show the equations for the model used in this paper
이 논문에서 사용한 Model 4 구조의 통계 모델 방정식(Model Equations)은 아래와 같이 도출됩니다. (통제변수(Covariate)들인 $C_k$ 스칼라 값들을 포함합니다.)

$$ M = i_M + aX + \sum_{k=1}^{4} d_{M,k} C_k + e_1 $$
$$ Y = i_Y + c'X + bM + \sum_{k=1}^{4} d_{Y,k} C_k + e_2 $$

여기서 $a \times b$ 가 간접효과(Indirect Effect)의 크기가 되며, $c'$는 직접효과(Direct Effect)를 의미합니다. $e_1, e_2$는 오차항(Error Term)입니다.

> **💬 심화 논의: $c = c' + a \times b$ 수학적 관계**
> 매개 모형에서는 모델링 에러의 측정 유무와 무관하게 **총효과($c$) = 직접효과($c'$) + 간접효과($a \times b$)** 라는 수식이 일반최소제곱법(OLS) 대수학적으로 완벽히 성립합니다 (결측치가 없는 팩트 데이터 기준). 즉, 직접 영향력을 행사하는 힘과 쿠션(M)을 맞고 행사하는 힘의 합산은 전체 타격량과 완전히 일치하게 됩니다.

## 6. List the covariates (통제변수) in this paper
본 연구에 쓰인 스칼라 값 형태의 통제변수(Covariates) 4가지와 논문에서 회귀 모형에 투입하기 위해 각 범주(Category)에 부여한 수치 코딩은 다음과 같습니다.
1. **성별 (Gender):** 남 = 1, 여 = 2
2. **학년 (Academic Year):** 1학년 = 1, 2학년 = 2, 3학년 = 3, 4학년 = 4
3. **한국어 능력 수준 / TOPIK 달성 (Korean Proficiency Level):** 초급 = 0, 중·고급 = 1
4. **주관적 가정 경제 수준 (Family Economic Status):** 하 = 1, 중 = 2, 상 = 3

## 7. 통제변수를 모두 고려한 equation을 풀어서 보여라 
실제 논문의 비표준화계수(Unstandardized Coefficient) $B$값(표 3 기준)을 수식에 매핑하여 통제변수들($C_1$: 성별, $C_2$: 학년, $C_3$: 한국어, $C_4$: 경제수준)을 모두 엮으면 다음과 같습니다.

* **경로 $a$ 모형 (M 구하기):**
$$ M = i_1 + 1.15 X + 3.72 C_1 + 0.68 C_2 + 2.26 C_3 - 1.01 C_4 + e_1 $$

* **경로 $b$ 및 직접효과 $c'$ 모형 (Y 구하기):**
$$ Y = i_2 - 1.76 X - 0.84 M - 13.14 C_1 - 1.19 C_2 - 2.44 C_3 + 2.50 C_4 + e_2 $$

## 8. 이 논문에서 통제변수를 위해 도입한 변수들의 기능을 설명하라.
> **💬 "통제변수는 노이즈(Noise)를 제거하는 변수입니다."**
> 이는 통제변수(Covariate)들의 영향을 대충 무시하겠다는 뜻이 결코 아닙니다. 오히려 **그 영향을 아주 철저하게 계산해서 제거해 내겠다**는 의미입니다. 교란변수(Confounding Variable)인 학년 효과나 성별 효과가 우리 연구의 핵심 주인공(X, Y 관계)인 것처럼 위장하지 못하도록, OLS 수학 모형이 이들의 영향치($d_1, d_2$ 등)를 Y에서 깎아내 버립니다(혼란변수 분리수거). 이렇게 통제해 놓고 나서야, 순수한 X$\rightarrow$M$\rightarrow$Y의 인과 메커니즘을 온전히 추정할 수 있습니다.

* **설계된 연속형 변환의 오류/위험성:** 저자들은 범주형 속성(Categorical property)인 학년을 1, 2, 3, 4로 분류해 넣고 연속형 변수(Continuous Variable)처럼 취급했습니다. 이는 1$\rightarrow$2학년으로 오를 때의 스트레스 변화량과, 3$\rightarrow$4학년으로 수료할 때의 스트레스 변화량이 수학적으로 정확히 "동일한 비율비례 보폭"을 가질 것이라고 무리하게 선형적 가정(Linearity Assumption)을 강제하게 됩니다.

## 9. 통제변수 도입을 위한 dummy coding 방식에 대해 설명 및 명시적 형태
범주형 변수(Categorical Variable)의 효과가 체계적으로 선형성(Linearity)을 띠지 않을 가능성이 높으므로, 각 집단을 온전히 분리해 주는 **더미 코딩(Dummy Coding)**이 필수적입니다.
만약 학년을 기존처럼 (1, 2, 3, 4)의 단일 숫자 모델로 넣으면 각 연차별 정적(+) 부적(-) 효과가 하나로 뭉개져 제2종 오류(Type II Error)에 직면할 염려가 큽니다.
이러한 은폐를 막기 위해 **$K$개의 범주 요소를 ($K-1$) 개의 더미 변수(Dummy Variable)로 분할**해야만 각 집단의 독특하고 비선형적(Non-linear)인 특징, 즉 고유 분산(Unique Variance)을 보존할 수 있습니다.

**'학년' 변수 명시적 예시 설명 (1학년 기준점):**
* $Dummy_2$: 학생이 2학년인가? 맞으면 $1$, 틀리면 $0$
* $Dummy_3$: 학생이 3학년인가? 맞으면 $1$, 틀리면 $0$
* $Dummy_4$: 학생이 4학년인가? 맞으면 $1$, 틀리면 $0$

## 10. Dummy coding 시 전체 회귀식 구성 빛 회귀계수의 의미
더미 변수(Dummy Variable)를 치환해 넣으면 문화적응스트레스($Y$) 모형의 수식은 이처럼 확장됩니다.
$$ Y = i + c'X + bM + d_2(Dummy_2) + d_3(Dummy_3) + d_4(Dummy_4) + e $$
* **회귀계수의 의미:** $i$값(절편 스칼라 값, Intercept) 속에는 더미가 모두 $0$인 기준 집단(Reference Group) 표본들의 베이스 수치가 내포됩니다. 
* **$d_2$ 계수의 의미:** 기준인 1학년들의 스트레스 점수 평균치에 비해 2학년 학생들이 추가로 느끼는 스트레스의 **부분적 평균 변화 차이(Marginal Difference/Shift)**가 됩니다. 방정식은 이를 독립적으로 계산해 주어 각 집단 고유의 카테고리 효과(Categorical Effect)를 정밀하게 잡아냅니다.

## 11. CTT 계열 방법론의 한계와 잠재변수(latent variable) 모델과의 차이
* **고전검사이론 (Classical Test Theory, CTT):** 사람들의 설문 측정값=기분이나 외부 노이즈 등을 포함한 관측 사실 전체를 단 한 지점의 "사실상 진짜 점수(True Score)"라고 무조건 믿어버리는 OLS 분석의 토대입니다. 측정 오차(Measurement Error)의 존재 가능성을 완전히 무시합니다.
* **잠재변수 모델 (Latent Variable Model, 추후 SEM):** 설문 조사는 언어장벽, 일시적 감정 등으로 항상 측정 오차(Measurement Error)가 있음을 상정합니다. 단순히 문항 합계를 내지 않고, 다수의 측정 지표(Indicators)를 통해 수면 아래 볼 수 없는 순수한 심리적 자아상인 **잠재변수(Latent Variable)**를 확률적으로 빚어냅니다. 이는 OLS에 비해 오염되지 않은 깔끔한 관계 매개추론을 제공합니다.

## 12. 베이지안 인퍼런스, 원-핫 인코딩, 잠재변수의 결합 및 장점

> **💬 심화 논의: 왜 머신러닝의 One-hot 대신 더미 코딩(K-1)을 쓰나요?**
> 기계학습에선 원-핫 인코딩을 널리 쓰지만, 고전 통계학의 일반최소제곱법(OLS) 회귀분석에서 $K$개의 원-핫 인코딩(One-Hot Encoding)과 상수(절편)항이 만나면 모형이 망가집니다. 원-핫 열 벡터의 총합이 상수항 열과 완전히 일치하게 되는데, 이를 **더미 변수의 함정(Dummy Variable Trap) 또는 완전 다중공선성(Perfect Multicollinearity)**이라고 부르며 행렬의 역행렬 추정을 불가능하게 만들어 모형을 그대로 붕괴시킵니다.

**결합 장점 (Bayesian + Latent + One-Hot):**
하지만 베이지안 추론(Bayesian Inference)은 빈도주의자(Frequentist)들의 OLS와 다르게 **사전분포(Prior Distribution)**나 **합-영 제약(Sum-to-zero Constraint)**을 수축 효과(Shrinkage Effect) 기반으로 심어줄 수 있습니다. 
즉, 매개변수의 식별성(Identifiability) 문제를 베이즈 룰로 잠재우기 때문에 붕괴 없이 $K$개의 원-핫 인코딩(One-Hot Encoding) 범주를 모조리 모형에 살려낼 수 있습니다. 
이 위에 **잠재변수(Latent Variable)**를 도입하게 되면, 측정 오차로 훼손되지 않은 순수한 심리학적 지표를, 마르코프 체인 몬테카를로(MCMC) 시뮬레이션을 통해 모든 집단의 분산 정보를 보존하며 **완벽한 확률적 사후분포(Posterior Distribution)**로 도출해내는 이상적인 연구 설계가 달성됩니다.

## 13. Show Stan code for this analysis.
```stan
data {
  int<lower=1> N; // Number of obs
  int<lower=1> K; // Items per scale
  matrix[N, K] x_items; // RSES
  matrix[N, K] m_items; // ISS
  matrix[N, K] y_items; // RSSIS
  
  // Categorical covariate data (1-based index for Stan array lookup)
  int<lower=1, upper=2> gender[N]; 
  int<lower=1, upper=4> year[N]; 
  int<lower=1, upper=2> topik[N]; // 1: 초급, 2: 중고급 (Adjusted from 0,1)
  int<lower=1, upper=3> eco[N]; 
}
parameters {
  vector[N] X; // 잠재변수 (Latent Variable)
  vector[N] M;
  vector[N] Y;
  
  vector<lower=0>[K] lambda_x;
  vector<lower=0>[K] lambda_m;
  vector<lower=0>[K] lambda_y;
  
  real a;
  real b;
  real cp;
  
  // 원-핫 인코딩 범주형 통제변수 (One-hot categorical grouped covariates)
  vector[2] d_gen_m;
  vector[4] d_year_m;
  vector[2] d_topik_m;
  vector[3] d_eco_m;

  vector[2] d_gen_y;
  vector[4] d_year_y;
  vector[2] d_topik_y;
  vector[3] d_eco_y;
  
  real<lower=0> sigma_m;
  real<lower=0> sigma_y;
  real<lower=0> sigma_items;
}
transformed parameters {
  // 모수 식별성(Identifiability) 제약을 위한 합-영 중앙화 (Sum-to-zero constraints)
  vector[2] d_gen_m_adj = d_gen_m - mean(d_gen_m);
  vector[4] d_year_m_adj = d_year_m - mean(d_year_m);
  vector[2] d_topik_m_adj = d_topik_m - mean(d_topik_m);
  vector[3] d_eco_m_adj = d_eco_m - mean(d_eco_m);
  
  vector[2] d_gen_y_adj = d_gen_y - mean(d_gen_y);
  vector[4] d_year_y_adj = d_year_y - mean(d_year_y);
  vector[2] d_topik_y_adj = d_topik_y - mean(d_topik_y);
  vector[3] d_eco_y_adj = d_eco_y - mean(d_eco_y);
}
model {
  // 베이지안 모형을 위한 정규화 사전분포 (Regularizing Priors)
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  cp ~ normal(0, 1);
  
  d_gen_m ~ normal(0, 1);
  d_year_m ~ normal(0, 1);
  d_topik_m ~ normal(0, 1);
  d_eco_m ~ normal(0, 1);
  
  d_gen_y ~ normal(0, 1);
  d_year_y ~ normal(0, 1);
  d_topik_y ~ normal(0, 1);
  d_eco_y ~ normal(0, 1);

  X ~ std_normal();
  
  // SEM 구조 모형 논리 (Structural Mediation Logic)
  for (i in 1:N) {
    M[i] ~ normal(a * X[i] + d_gen_m_adj[gender[i]] + d_year_m_adj[year[i]] + d_topik_m_adj[topik[i]] + d_eco_m_adj[eco[i]], sigma_m);
    Y[i] ~ normal(cp * X[i] + b * M[i] + d_gen_y_adj[gender[i]] + d_year_y_adj[year[i]] + d_topik_y_adj[topik[i]] + d_eco_y_adj[eco[i]], sigma_y);
  }
  
  // 측정 모형 산출 (Measurement model computation)
  for (k in 1:K) {
    x_items[, k] ~ normal(lambda_x[k] * X, sigma_items);
    m_items[, k] ~ normal(lambda_m[k] * M, sigma_items);
    y_items[, k] ~ normal(lambda_y[k] * Y, sigma_items);
  }
}
generated quantities {
  // 간접효과 사후 분포 측정 (Posterior draws for indirect effect)
  real indirect_effect = a * b; 
}
```

## 14. 파이썬을 사용한 경로 다이어그램(Path Diagram) 생성 코드
R(lavaan)이나 Mermaid를 대체하여, 파이썬의 `graphviz` 패키지를 통해 논문의 매개 모형 경로 다이어그램(Path Diagram)을 시각화하는 코드입니다.

```python
import graphviz

# 다이렉트 그래프 객체 생성 (왼쪽에서 오른쪽으로 렌더링)
dot = graphviz.Digraph(comment='Mediation Model', format='png')
dot.attr(rankdir='LR', size='10,6')

# 핵심 변수 노드(Node) 추가
dot.node('X', 'Self-Esteem\n(X)', shape='box', style='filled', fillcolor='#f9d0c4')
dot.node('M', 'Intercultural Sensitivity\n(M)', shape='box', style='filled', fillcolor='#d0f9c4')
dot.node('Y', 'Acculturative Stress\n(Y)', shape='box', style='filled', fillcolor='#c4d0f9')

# 통제변수 클러스터 생성
with dot.subgraph(name='cluster_covariates') as c:
    c.attr(label='Covariates (Controls)', style='dashed', color='grey')
    c.node('C1', 'Gender', shape='ellipse')
    c.node('C2', 'Academic Year', shape='ellipse')
    c.node('C3', 'TOPIK Status', shape='ellipse')
    c.node('C4', 'Economic Level', shape='ellipse')

# 메인 인과 경로(Edges) 추가
dot.edge('X', 'M', label=' Path a', color='blue')
dot.edge('M', 'Y', label=' Path b', color='blue')
dot.edge('X', 'Y', label=' Path c\' (Direct)', color='red')

# 통제변수 경로 추가 (통제변수가 M과 Y에 미치는 영향)
covariates = ['C1', 'C2', 'C3', 'C4']
for cov in covariates:
    dot.edge(cov, 'M', style='dashed', color='grey')
    dot.edge(cov, 'Y', style='dashed', color='grey')

# 그래프 렌더링 및 파일 저장
dot.render('kim2025_path_diagram', view=False)
print("Path diagram successfully saved as 'kim2025_path_diagram.png'.")
```

## 16. Explain why this is called Bayesian SEM.
일반적인 **구조방정식 모델링(Structural Equation Modeling, SEM)**은 파라미터를 점 추정치(Point Estimation, 통상 최대우도추정(Maximum Likelihood Estimation, MLE)) 방식으로 계산합니다. 하지만 이 프로세스를 포기하고 모수들을 확률로 탐색하는 **베이지안 추론(Bayesian Inference)** 시뮬레이션 기법을 결합하여 구워냈기 때문에 **BSEM(Bayesian SEM)**이라는 명칭이 붙게 되었습니다.

## 17. Explain why BSEM should be chosen over CTT for this analysis.
1. **잠재변수를 통한 쇠퇴 보정(Correction for Attenuation):** 고전검사이론(CTT)처럼 원시 측정 데이터를 그대로 투입하면 각 문항의 측정 오류(Measurement Errors)가 간접효과($a \times b$)를 수학적으로 과소/과대 측정(Bias)할 위험이 큽니다. BSEM의 구조방정식을 도입하면 내적인 오류가 배제된 진짜 변동성을 잡아낼 수 있게 됩니다.
2. **비선형성(Non-linear) 및 다중공선성(Multicollinearity)의 유연한 제어:** 더미 코딩과 원-핫 인코딩(One-hot Encoding) 등 비선형적 묶음 효과를 수렴시킬 때 고전적 고정점 회귀가 아닌 분포형 사전분포(Prior Distribution) 제약 구조를 이용하여 모형을 붕괴시키지 않고 강건(Robust)하게 추론해 냅니다.
3. **가장 진실한 간접효과의 신뢰성:** 빈도주의/CTT 계열의 비대칭 부트스트래핑(Bootstrapping) 시뮬레이션보다 근본적으로 뛰어납니다. BSEM은 무수히 굴려지는 마르코프 체인(MCMC) 사후분포에서 직접 값을 추출하므로, 정교한 95% **최고사후밀도구간(Highest Posterior Density Interval, HPDI)** 매개변수 파라미터 신뢰구간(CI)을 즉각적으로 획득할 수 있습니다.
