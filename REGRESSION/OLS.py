import statsmodels.api as sm
from statsmodels.api import OLS
import pandas as pd
from sklearn.datasets import load_diabetes


diabetes = load_diabetes()

dfx = diabetes.data
feature_names =['constant'] + diabetes.feature_names
dfx = sm.add_constant(dfx)
dfx = pd.DataFrame(dfx, columns=feature_names)
dfy = pd.DataFrame(diabetes.target, columns=['target'])

df = pd.concat([dfx, dfy], axis=1)

model = OLS(dfx, dfy)

result = model.fit()
