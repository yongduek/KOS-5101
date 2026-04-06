import json
import pandas as pd
import numpy as np

# Load data and compute
df_rses = pd.read_csv('rses_simulated.csv')
df_rssis = pd.read_csv('rssis_simulated.csv')
df_iss = pd.read_csv('iss_simulated.csv')

df_total = pd.DataFrame({
    'RSES': df_rses.sum(axis=1),
    'RSSIS': df_rssis.sum(axis=1),
    'ISS': df_iss.sum(axis=1)
})

print("=== Descriptive Statistics ===")
print(df_total.describe().loc[['mean', 'std']].round(2))
print("\n=== Correlations ===")
print(df_total.corr().round(2))

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 김연하 외 (2025) 연구 데이터 시뮬레이션 기술 통계 검증\n",
    "본 노트북은 파이썬 시뮬레이터(`simulate_data.py`)를 통해 생성된 가상 응답 데이터셋(CSV) 3종을 분석하여, 논문에 제시된 기술통계량 및 피어슨 상관계수 등 핵심 수치들이 소수점 둘째 자리 내에서 얼마나 수치적으로 완벽하게 재현되었는지 검증합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# 1. 데이터 불러오기\n",
    "df_rses = pd.read_csv('rses_simulated.csv')\n",
    "df_rssis = pd.read_csv('rssis_simulated.csv')\n",
    "df_iss = pd.read_csv('iss_simulated.csv')\n",
    "\n",
    "# 각 척도의 총합 산출\n",
    "df_total = pd.DataFrame({\n",
    "    'RSES (자아존중감)': df_rses.sum(axis=1),\n",
    "    'RSSIS/ASSIS (문화적응스트레스)': df_rssis.sum(axis=1),\n",
    "    'ISS (상호문화감수성)': df_iss.sum(axis=1)\n",
    "})\n",
    "\n",
    "print(f\"데이터 샘플 수 (N): {len(df_total)}\")\n",
    "df_total.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. 기술 통계량 (Mean, SD) 비교\n",
    "print(\"=========================================\")\n",
    "print(\"[논문 원문 목표 스펙 (Target)]\")\n",
    "print(\"RSES: Mean = 29.87, SD = 5.24\")\n",
    "print(\"RSSIS: Mean = 81.84, SD = 31.49\")\n",
    "print(\"ISS: Mean = 86.10, SD = 12.27\")\n",
    "print(\"=========================================\")\n",
    "print(\"[데이터프레임 관측치 산출 (Simulated)]\")\n",
    "print(df_total.describe().loc[['mean', 'std']].round(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. 상관관계 (Pearson Correlation) 비교\n",
    "print(\"=========================================\")\n",
    "print(\"[논문 원문 타겟 상관계수 (r)]\")\n",
    "print(\"RSES <-> ISS: 0.46\")\n",
    "print(\"RSES <-> RSSIS: -0.41\")\n",
    "print(\"ISS <-> RSSIS: -0.49\")\n",
    "print(\"=========================================\")\n",
    "print(\"[시뮬레이션 데이터 내부 상관계수]\")\n",
    "print(df_total.corr().round(2))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("gemini_kimm2025_simulate.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("\n[SUCCESS] Notebook gemini_kimm2025_simulate.ipynb created.")
