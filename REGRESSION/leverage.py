from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

plt.rc('font', family='Malgun Gothic')

# 100개의 데이터 생성
X0, y, coef = make_regression(n_samples=100, n_features=1, noise=20,
                              coef=True, random_state=1)

# 레버리지가 높은 가상의 데이터를 추가
data_100 = (4, 300)
data_101 = (3, 150)
X0 = np.vstack([X0, np.array([data_100[:1], data_101[:1]])])
X = sm.add_constant(X0)
y = np.hstack([y, [data_100[1], data_101[1]]])

model = sm.OLS(pd.DataFrame(y), pd.DataFrame(X))
result = model.fit()
print(result.summary())

influence = result.get_influence()
hat = influence.hat_matrix_diag
print(f'레버리지 합 : {hat.sum()}')

# plt.scatter(X0, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("가상의 회귀분석용 데이터")
# plt.show()

influence = result.get_influence()
hat = influence.hat_matrix_diag

plt.figure(figsize=(10, 2))
plt.stem(hat)
plt.axhline(0.02, c="g", ls="--")
plt.title("각 데이터의 레버리지 값")
plt.show()