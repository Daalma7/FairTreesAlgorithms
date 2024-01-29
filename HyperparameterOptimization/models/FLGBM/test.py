import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
import seaborn as sns

np.random.seed(7)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)



df2 = df.copy()



print(df2.columns)

df2 = df2.loc[:, ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension']]

df = df.loc[:, ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension']]


df2['randNumCol_1'] = np.random.random(df2.shape[0])
df2['randNumCol_2'] = np.random.random(df2.shape[0])
df2['randNumCol_3'] = np.random.random(df2.shape[0])
df2['randNumCol_4'] = np.random.random(df2.shape[0])
df2['randNumCol_5'] = np.random.random(df2.shape[0])

print("----")
print(df.columns)
print(df2.columns)
print("----")

X_train, X_test, y_train, y_test = train_test_split(df, data.target, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

lgbm_params = {
    'objective': 'binary',
    'random_seed': 0,
    'verbose': -1
    }
model = lgb.train(lgbm_params, 
                  lgb_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred>0.5))
print(roc_auc_score(y_test, y_pred))

# >> 0.972027972027972

X_train, X_test, y_train, y_test = train_test_split(df2, data.target, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

lgbm_params = {
    'objective': 'binary',
    'random_seed': 0,
    'verbose': -1
    }
model = lgb.train(lgbm_params, 
                  lgb_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred>0.5))
print(roc_auc_score(y_test, y_pred))
