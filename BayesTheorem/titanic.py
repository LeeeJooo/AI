import pandas as pd
import numpy as np
import random

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB  # 가우시안 분포를 따르는 나이브즈 확률..
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

print(f'[train_df.columns.values] : \n{train_df.columns.values}\n')
print(f'[test_df.columns.values] : \n{test_df.columns.values}\n')

print('[train_df head]')
print(train_df.head(), end='\n\n')

print('[train_df describe]')
print(train_df.describe(), end='\n\n')



train_df = train_df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)

train_df['Sex'] = train_df['Sex'].map({'female':1, 'male':0}).astype(int)
test_df['Sex'] = test_df['Sex'].map({'female':1, 'male':0}).astype(int)


guess_ages = np.zeros((2,3))
combine = [train_df, test_df]
for dataset in combine:
    for i in range(2):
        for j in range(3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j-1] = int(age_guess/0.5+0.5)*0.5   # 0.5 단위로 반올림
    
    for i in range(2):
        for j in range(3):
            dataset.loc[
                (dataset.Age.isnull()) & (dataset.Sex==i) & (dataset.Pclass==j+1) ,\
                'Age'           
            ] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
test_df['AgeBand'] = pd.cut(train_df['Age'], 5)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 80), 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
test_df = test_df.drop(['AgeBand'], axis=1)

train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].dropna().median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].dropna().median())
# train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
# test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

print(train_df['Fare'].isnull().sum())  # NaN 개수 확인
print(test_df['Fare'].isnull().sum())

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)

combine = [train_df, test_df]

# for dataset in combine:
#     dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

X_train = train_df.drop(['Survived', 'Embarked'], axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop(['Embarked'], axis=1)

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# DataFrame 변환 (각 클래스에 대한 확률)
probs = gaussian.predict_proba(X_train)
probs_df = pd.DataFrame(probs, columns=["dead", "survived"])

# 히트맵 시각화
plt.figure(figsize=(8,6))
sns.heatmap(probs_df.head(20), annot=True, cmap="coolwarm", cbar=True)
plt.xlabel("Survived")
plt.ylabel("Samples")
plt.title("GaussianNB Probability Heatmap")
plt.legend()
plt.show()