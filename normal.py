import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 파라미터 설정 ---
mu = 75       # 평균 (Mean)
sigma = 10    # 표준편차 (Standard Deviation)
size = 5000   # 샘플 개수 (Sample size)

# 1. Sampling: 난수 생성 및 샘플링
rng = np.random.default_rng(seed=42)
samples = rng.normal(loc=mu, scale=sigma, size=size)

# 2. PDF 라인 계산: 이론적 곡선 도출
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf = norm.pdf(x, mu, sigma)

# 3. Pyplot 그리기 및 파일 저장
plt.figure(figsize=(10, 6))

# 히스토그램 (실제 데이터 분포)
plt.hist(samples, bins=50, density=True, alpha=0.5, color='skyblue', edgecolor='gray')

# PDF 라인 (이론적 수치)
plt.plot(x, pdf, color='red', linewidth=2)

# 스타일링 및 저장
plt.title(f'Normal Distribution ($\mu={mu}, \sigma={sigma}$)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# WSL에서는 plt.show() 대신 반드시 savefig를 사용합니다.
plt.savefig('normal_distribution_result.png', dpi=300)
print("성공: 'normal_distribution_result.png' 파일로 저장되었습니다.")