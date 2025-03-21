from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

california_housing = fetch_california_housing()

dfx = pd.DataFrame(california_housing.data, columns = california_housing.feature_names)
dfy = pd.DataFrame(california_housing.target, columns = california_housing.target_names)

print(f'독립 변수 : {califo nia_housing.feature_names}')
# 독립 변수 : ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(f'종속 변수 : {california_housing.target_names}')
# 종속 변수 : ['MedHouseVal']
df = pd.concat([dfx, dfy], axis=1)
print(df.tail())

sns.pairplot(df[california_housing.feature_names + california_housing.target_names])
plt.show()
