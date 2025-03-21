import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('car_evaluation.csv')

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
df.columns = col_names

X = df.drop(['label'], axis=1)
Y = df['label']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)

clt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clt_gini.fit(X_train, Y_train)

y_pred_gini = clt_gini.predict(X_test)
y_pred_train_gini = clt_gini.predict(X_train)

print('Training set score: {:.4f}'.format(clt_gini.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(clt_gini.score(X_test, Y_test)))