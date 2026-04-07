# Bayesian Structural Equation Modeling (B-SEM) 분석 가이드

본 저장소는 논문(김연하 외, 2025)의 "자아존중감-상호문화감수성-문화적응스트레스" 매개모형을 **다요인(Multi-factor) 기반의 베이지언 구조방정식**으로 확장하여 재현하고 평가하는 파이썬 코드 및 주피터 노트북을 포함하고 있습니다.

## 📁 파일 구성
1. `kim2025_multifactor_simulate.py`
   - **역할**: 2차 확인적 요인분석(Second-order CFA) 구조를 적용하여 진점수(True score) 기반의 시뮬레이션 데이터를 생성합니다. 
   - **특징**: ISS(5요인), ASSIS(7요인)의 실제 측정도구 구조를 반영하며, B-SEM의 계산 안정성을 위해 문항 꾸러미(Item Parceling) 처리된 데이터를 산출합니다.
2. `kim2025_bsem_analysis.ipynb`
   - **역할**: 생성된 데이터를 활용해 Stan(cmdstanpy) 기반의 B-SEM 분석을 수행하는 노트북입니다.
   - **특징**: 네트워크 다이어그램 시각화, 잠재변수 간 경로계수(MCMC) 추정, 사후 분포 시각화, 그리고 '데이터 생성 시 설정한 참값(True Value)과의 비교 평가' 기능이 포함되어 있습니다.

## 🚀 실행 방법

**1단계: 필수 라이브러리 설치**
```bash
pip install pandas numpy matplotlib networkx cmdstanpy arviz
```