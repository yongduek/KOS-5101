import numpy as np

# 1. 데이터 로드 (상대 경로)
try:
    joint_prob = np.genfromtxt('./26-1/prob.csv', delimiter=',')
except FileNotFoundError:
    print("Error: 파일을 찾을 수 없습니다.")
    exit()

print(type(joint_prob))
print("loaded numpy array:")
print(joint_prob)
print(joint_prob.sum())

rows, cols = joint_prob.shape

# 2. 주변 확률 계산
marginal_x = np.sum(joint_prob, axis=1)
marginal_y = np.sum(joint_prob, axis=0)

print("Marginal Distribution_x :", marginal_x)
print("Marginal Distribution_y :", marginal_y)

# 3. 조건부 확률 계산
# P(Y|X) 계산 (각 행을 marginal_x로 나눔)
cond_y_given_x = joint_prob / marginal_x[:, np.newaxis]

# P(X|Y) 계산 (각 열을 marginal_y로 나눔)
cond_x_given_y = joint_prob / marginal_y[np.newaxis, :]

# 4. 노테이션을 포함한 출력
print("--- Conditional Distribution P(Y|X) ---")
for i in range(rows):
    for j in range(cols):
        # f-string을 사용하여 (yj|xi) 형식으로 출력
        print(f"P(y{j+1}|x{i+1}) = {cond_y_given_x[i, j]:.4f}", end="  ")
    print()  # 행 바꿈

print("\n--- Conditional Distribution P(X|Y) ---")
for j in range(cols):
    for i in range(rows):
        # f-string을 사용하여 (xi|yj) 형식으로 출력
        print(f"P(x{i+1}|y{j+1}) = {cond_x_given_y[i, j]:.4f}", end="  ")
    print()  # 열 바꿈

# 과제: 6.1의 b. conditional distributions p(x|Y=y1) and P(y|X=x3)