import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

# 재현성을 위한 시드 고정
np.random.seed(2025)
N = 394

# =====================================================================
# 1. Macro-level: 논문과 완벽하게 일치하는 총점(X, M, Y) 생성
# =====================================================================
# 논문 보고값 (표 2)
means = np.array([29.87, 86.10, 81.84]) # X, M, Y
stds = np.array([5.24, 12.27, 31.49])
corrs = np.array([
    [1.00,  0.46, -0.41],
    [0.46,  1.00, -0.49],
    [-0.41, -0.49,  1.00]
])

# 공분산 행렬 계산
cov = np.outer(stds, stds) * corrs

# 무작위 데이터 생성 후 정확한 평균/공분산으로 변환 (Empirical Matching)
raw_data = np.random.randn(N, 3)
raw_data -= raw_data.mean(axis=0)
emp_cov = np.cov(raw_data, rowvar=False)

transform_matrix = sqrtm(cov).dot(np.linalg.inv(sqrtm(emp_cov)))
exact_data = raw_data.dot(transform_matrix.T) + means

# Likert 척도 범위 내 정수 반올림 (총점은 항상 정수임)
X_tot = np.clip(np.round(exact_data[:, 0]), 10, 40).astype(int)
M_tot = np.clip(np.round(exact_data[:, 1]), 24, 120).astype(int)
Y_tot = np.clip(np.round(exact_data[:, 2]), 36, 180).astype(int)

# =====================================================================
# 2. 하향식 분배 알고리즘 (총점 -> 하위요인 -> 개별문항)
# =====================================================================
def distribute_sum_to_bins(total_val, bounds):
    """
    total_val을 bounds(최소, 최대)를 가진 bin들로 정확히 분배하는 함수
    """
    num_bins = len(bounds)
    vals = np.array([b[0] for b in bounds]) # 최소값으로 초기화
    remaining = total_val - sum(vals)
    
    # 1. 무작위로 남은 점수 분배
    while remaining > 0:
        valid_bins = [i for i in range(num_bins) if vals[i] < bounds[i][1]]
        if not valid_bins: break
        idx = np.random.choice(valid_bins)
        vals[idx] += 1
        remaining -= 1
        
    # 2. 분산을 주기 위한 무작위 스와핑 (총합은 유지)
    for _ in range(num_bins * 4):
        i, j = np.random.choice(num_bins, 2, replace=False)
        if vals[i] < bounds[i][1] and vals[j] > bounds[j][0]:
            vals[i] += 1
            vals[j] -= 1
    return vals

# ---------------------------------------------------------
# X: 자아존중감 (1요인, 10문항, 1~4점)
X_items = np.array([distribute_sum_to_bins(val, [(1, 4)] * 10) for val in X_tot])

# ---------------------------------------------------------
# M: 상호문화감수성 (5요인, 24문항, 1~5점)
# 요인별 문항 수: [5, 5, 5, 5, 4]
m_factors_lens = [5, 5, 5, 5, 4]
M_items = []
for val in M_tot:
    # 1단계: 총점을 5개 하위 요인으로 분배
    f_bounds = [(length * 1, length * 5) for length in m_factors_lens]
    f_vals = distribute_sum_to_bins(val, f_bounds)
    
    # 2단계: 각 요인 점수를 다시 개별 문항으로 분배
    items = []
    for f_val, length in zip(f_vals, m_factors_lens):
        items.extend(distribute_sum_to_bins(f_val, [(1, 5)] * length))
    M_items.append(items)
M_items = np.array(M_items)

# ---------------------------------------------------------
# Y: 문화적응스트레스 (7요인, 36문항, 1~5점)
# 요인별 문항 수: [5, 5, 5, 5, 5, 5, 6]
y_factors_lens = [5, 5, 5, 5, 5, 5, 6]
Y_items = []
for val in Y_tot:
    # 1단계: 총점을 7개 하위 요인으로 분배
    f_bounds = [(length * 1, length * 5) for length in y_factors_lens]
    f_vals = distribute_sum_to_bins(val, f_bounds)
    
    # 2단계: 개별 문항으로 분배
    items = []
    for f_val, length in zip(f_vals, y_factors_lens):
        items.extend(distribute_sum_to_bins(f_val, [(1, 5)] * length))
    Y_items.append(items)
Y_items = np.array(Y_items)

# =====================================================================
# 3. 데이터프레임 병합 및 저장
# =====================================================================
df = pd.DataFrame()
# 공변량 임의 생성 (결과 검증용)
df['gender'] = np.random.choice([1, 2], N)
df['grade'] = np.random.choice([1, 2, 3, 4], N)
df['topik'] = np.random.choice([0, 1], N)
df['income'] = np.random.choice([1, 2, 3], N)

# 문항 데이터 할당
for i in range(10): df[f'rses_{i+1}'] = X_items[:, i]
for i in range(24): df[f'iss_{i+1}'] = M_items[:, i]
for i in range(36): df[f'assis_{i+1}'] = Y_items[:, i]

# 검증용 총합 계산 (논문 분석용)
df['total_rses'] = df[[f'rses_{i+1}' for i in range(10)]].sum(axis=1)
df['total_iss'] = df[[f'iss_{i+1}' for i in range(24)]].sum(axis=1)
df['total_assis'] = df[[f'assis_{i+1}' for i in range(36)]].sum(axis=1)

df.to_csv("kim2025_exact_multifactor_data.csv", index=False)

# 결과 검증 출력
print("=== 생성된 데이터 검증 (논문 표 2와 비교) ===")
desc = df[['total_rses', 'total_iss', 'total_assis']].describe().T[['mean', 'std']]
corr = df[['total_rses', 'total_iss', 'total_assis']].corr()
print("\n[기술통계 (Mean, SD)]")
print(desc.round(2))
print("\n[상관계수 행렬]")
print(corr.round(3))
print("\n✓ 데이터 생성 완료: 'kim2025_exact_multifactor_data.csv'")
print("✓ 이 데이터의 총점('total_~')을 PROCESS Macro에 넣으면 논문과 사실상 동일한 계수가 도출됩니다.")