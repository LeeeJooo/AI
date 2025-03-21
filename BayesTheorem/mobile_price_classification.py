import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from pprint import pprint

df = pd.read_csv('./dataset/mobile_price/train.csv')

continuous_var = []
binary_var = []
for col in df.columns.values:
    if df[col].value_counts().count() > 2 :
        continuous_var.append(col)
    else :
        binary_var.append(col)

target_var = 'price_range'
continuous_var.remove('price_range')

X = df.drop(target_var, axis=1)
y = df[target_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_continuous = X_train[continuous_var]
X_train_binary = X_train[binary_var]

X_test_continuous = X_test[continuous_var]
X_test_binary = X_test[binary_var]

gnb = GaussianNB()
bnb = BernoulliNB()

gmodel = gnb.fit(X_train_continuous, y_train)
bmodel = bnb.fit(X_train_binary, y_train)

g_log_likelihood = gmodel._joint_log_likelihood(X_test_continuous)
b_log_likelihood = bmodel._joint_log_likelihood(X_test_binary)
log_likelihood = g_log_likelihood + b_log_likelihood

y_classes = y_train.unique()
y_classes = np.sort(y_classes)

g_y_pred = y_classes[np.argmax(g_log_likelihood, axis=1)]
b_y_pred = y_classes[np.argmax(b_log_likelihood, axis=1)]
mix_y_pred = y_classes[np.argmax(log_likelihood, axis=1)]

print(f'GAUSSIAN ACC : {int((g_y_pred==y_test).sum()/len(y_test)*100)} %')
print(f'BERNOULLI ACC : {int((b_y_pred==y_test).sum()/len(y_test)*100)} %')
print(f'MIX ACC : {int((mix_y_pred==y_test).sum()/len(y_test)*100)} %')

print()

y_unique = y.unique()
y_unique = np.sort(y_unique)

confusion_matrix = np.zeros((y_unique.shape[0], y_unique.shape[0]))
for ans in y_unique:
    for i in y_unique:
        confusion_matrix[ans][i] = ((y_test==ans) & (mix_y_pred==i)).sum()

gaussian_confusion_matrix = np.zeros((y_unique.shape[0], y_unique.shape[0]))
for ans in y_unique:
    for i in y_unique:
        gaussian_confusion_matrix[ans][i] = ((y_test==ans) & (g_y_pred==i)).sum()

bernoulli_confusion_matrix = np.zeros((y_unique.shape[0], y_unique.shape[0]))
for ans in y_unique:
    for i in y_unique:
        bernoulli_confusion_matrix[ans][i] = ((y_test==ans) & (b_y_pred==i)).sum()

print('confusion matrix')
pprint(confusion_matrix)
print()

print('gaussian_confusion_matrix')
pprint(gaussian_confusion_matrix)
print()

print('bernoulli_confusion_matrix')
pprint(bernoulli_confusion_matrix)
print()
        
# recall
recall_mixed = np.zeros(y_unique.shape[0])
recall_gaussian = np.zeros(y_unique.shape[0])
recall_bernoulli = np.zeros(y_unique.shape[0])

for ans in y_unique:
    recall_mixed[ans] = confusion_matrix[ans][ans]/confusion_matrix[ans, :].sum()
    recall_gaussian[ans] = gaussian_confusion_matrix[ans][ans]/gaussian_confusion_matrix[ans, :].sum()
    recall_bernoulli[ans] = bernoulli_confusion_matrix[ans][ans]/bernoulli_confusion_matrix[ans, :].sum()

# precision
precision_mixed = np.zeros(y_unique.shape[0])
precision_gaussian = np.zeros(y_unique.shape[0])
precision_bernoulli = np.zeros(y_unique.shape[0])

for ans in y_unique:
    precision_mixed[ans] = confusion_matrix[ans][ans]/confusion_matrix[:,ans].sum()
    precision_gaussian[ans] = gaussian_confusion_matrix[ans][ans]/gaussian_confusion_matrix[:,ans].sum()
    precision_bernoulli[ans] = bernoulli_confusion_matrix[ans][ans]/bernoulli_confusion_matrix[:,ans].sum()

recall_mixed_ = recall_mixed.sum()/recall_mixed.shape[0]
precision_mixed_ = precision_mixed.sum()/precision_mixed.shape[0]

recall_gaussian_ = recall_gaussian.sum()/recall_gaussian.shape[0]
precision_gaussian_ = precision_gaussian.sum()/precision_gaussian.shape[0]

recall_bernoulli_ = recall_bernoulli.sum()/recall_bernoulli.shape[0]
precision_bernoulli_ = precision_bernoulli.sum()/precision_bernoulli.shape[0]

print(f'recall_mixed : {recall_mixed_:.2f}, recall_gaussian : {recall_gaussian_:.2f}, recall_bernoulli : {recall_bernoulli_:.2f}')
print()

print(f'precision_mixed : {precision_mixed_:.2f}, precision_gaussian : {precision_gaussian_:.2f}, precision_bernoulli : {precision_bernoulli_:.2f}')