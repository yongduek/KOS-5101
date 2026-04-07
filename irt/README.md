# IRT Jupyter Notebooks

**Item Response Theory (IRT) 모델 구현 및 시각화**

이 컬렉션은 주요 IRT 모델을 Python과 Stan(cmdstanpy)으로 구현한 18개의 Jupyter 노트북으로 구성됩니다. 각 노트북은 모델 이론 설명, Stan 코드, 시뮬레이션, 파라미터 추정, 시각화를 포함합니다.

---

## 노트북 목록

### 이분 모델 (Dichotomous Models)

| 파일 | 모델 | 설명 |
|------|------|------|
| `IRT_D1_Rasch_1PL.ipynb` | Rasch / 1PL | 단일 난이도 파라미터, 로그잇 단위 |
| `IRT_D2_2PL.ipynb` | 2PL | 난이도 + 변별도 |
| `IRT_D3_3PL.ipynb` | 3PL | 난이도 + 변별도 + 추측도(guessing) |

### 다분 모델 (Polytomous Models)

| 파일 | 모델 | 설명 |
|------|------|------|
| `IRT_P1_GRM.ipynb` | GRM | Graded Response Model — 누적확률 기반 |
| `IRT_P2_PCM.ipynb` | PCM | Partial Credit Model — 인접범주 로그오즈 |
| `IRT_P3_GPCM.ipynb` | GPCM | Generalized PCM — PCM + 변별도 |
| `IRT_P4_RSM.ipynb` | RSM | Rating Scale Model — 공통 단계난이도 |
| `IRT_P5_NRM.ipynb` | NRM | Nominal Response Model — 명목 범주 |
| `IRT_P6_Sequential.ipynb` | Sequential | 순차적(단계별) 반응 모델 |
| `IRT_P7_MCM.ipynb` | MCM | Multiple-Choice Model — 오답지 분석 |

### 다국면 모델 (Multi-Facet Models)

| 파일 | 모델 | 설명 |
|------|------|------|
| `IRT_MF1_MFRM.ipynb` | MFRM | Many-Facet Rasch Model |
| `IRT_MF2_MFPCM.ipynb` | MF-PCM | Many-Facet PCM |
| `IRT_MF3_MFRSM.ipynb` | MF-RSM | Many-Facet RSM |
| `IRT_MF4_MFGPCM.ipynb` | MF-GPCM | Many-Facet GPCM |
| `IRT_MF5_MFGRM.ipynb` | MF-GRM | Many-Facet GRM |
| `IRT_MF6_HRM.ipynb` | HRM | Hierarchical Rater Model |
| `IRT_MF7_LLTM_Facets.ipynb` | LLTM+Facets | Linear Logistic Test Model with Facets |

### 모델 비교

| 파일 | 설명 |
|------|------|
| `IRT_models.ipynb` | 전체 모델 개요, PCM vs GRM 비교, 모델 선택 가이드 |

---

## 실행 방법

### 빠른 실행 (권장)

**Windows:**
```
exe_notebooks.bat
```

**macOS / Linux:**
```bash
bash exe_notebooks.sh
```

특정 노트북만 실행하거나 타임아웃을 조정하려면:

```bash
# 특정 노트북만 실행
bash exe_notebooks.sh IRT_P2_PCM.ipynb

# 셀 타임아웃을 180초로 설정 (Stan 추정 시간이 긴 경우)
bash exe_notebooks.sh --timeout 180
```

### Jupyter Lab / Notebook 사용

```bash
jupyter lab
# 또는
jupyter notebook
```

브라우저에서 원하는 노트북을 열고 **Kernel → Restart & Run All** 선택.

---

## 설치 요구사항

### 필수

```bash
pip install numpy scipy matplotlib pandas jupyter nbconvert
```

### Stan 기반 베이지안 추정 (선택)

Stan이 없어도 노트북은 실행됩니다. Stan이 없을 경우 시뮬레이션 데이터를 기반으로 한 근사 결과가 자동으로 사용됩니다.

```bash
# cmdstanpy 설치
pip install cmdstanpy

# CmdStan 설치 (한 번만 실행)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

또는 conda를 사용하는 경우:

```bash
conda install -c conda-forge cmdstanpy
```

### 한국어 폰트 (선택)

그래프의 한국어 레이블을 올바르게 표시하려면 시스템에 한국어 폰트가 필요합니다.

- **macOS**: 기본 설치됨 (Apple SD Gothic Neo)
- **Windows**: 기본 설치됨 (맑은 고딕)
- **Linux**: `sudo apt-get install fonts-nanum` 또는 `fonts-droid-fallback`

---

## 내장 실행기 (`run_notebooks.py`)

`run_notebooks.py`는 Jupyter/nbconvert 없이도 노트북을 실행할 수 있는 내장 실행기입니다. `exe_notebooks.sh` / `exe_notebooks.bat`이 자동으로 이 파일을 사용합니다.

직접 실행도 가능합니다:

```bash
python run_notebooks.py                      # 모든 노트북 실행
python run_notebooks.py IRT_P2_PCM.ipynb    # 특정 노트북만
python run_notebooks.py --timeout 180       # 타임아웃 설정
python run_notebooks.py --quiet             # 상세 출력 억제
```

---

## 주요 파라미터 기호 정리

| 기호 | 의미 |
|------|------|
| θ (theta) | 피험자 능력 (latent trait) |
| a | 변별도 (discrimination) |
| b, δ | 난이도 / 단계난이도 (difficulty / threshold) |
| c | 추측도 (guessing, lower asymptote) |
| K | 반응 범주 수 |
| J | 피험자 수 |
| I | 문항 수 |
| φ (phi) | 채점자 엄격성 (rater severity) |

---

## 참고사항

- 각 노트북은 **독립적으로 실행** 가능합니다.
- Stan 추정 셀은 `if STAN_AVAILABLE:` 블록으로 보호되어 있어, cmdstanpy 없이도 시각화 결과를 확인할 수 있습니다.
- 노트북 실행 후 결과는 `.ipynb` 파일에 저장되므로, Jupyter 없이 GitHub이나 VS Code에서도 결과를 확인할 수 있습니다.
