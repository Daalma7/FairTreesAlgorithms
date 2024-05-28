import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import FLGBM
import time

np.random.seed(7)

dataname = 'adult'
protname = 'Race'
X_train = pd.read_csv(f"../../../../datasets/data/train_val_test_standard/{dataname}/{dataname}_{protname}_train_seed_100.csv")
print(X_train)
y_train = X_train.iloc[:,-1]
X_train = X_train.iloc[:, :-1]

X_val = pd.read_csv(f"../../../../datasets/data/train_val_test_standard/{dataname}/{dataname}_{protname}_val_seed_100.csv")
print(X_val)
y_val = X_val.iloc[:,-1]
X_val = X_val.iloc[:, :-1]

X_test = pd.read_csv(f"../../../../datasets/data/train_val_test_standard/{dataname}/{dataname}_{protname}_test_seed_100.csv")
print(X_test)
y_test = X_test.iloc[:,-1]
X_test = X_test.iloc[:, :-1]


lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

lgbm_params = {
    'objective': 'binary',
    'random_seed': 0,
    'verbose': -1
    }
start = time.process_time()
model = lgb.train(lgbm_params, 
                  lgb_train)
end = time.process_time()
print("Tiempo LightGBM normal: ", end-start)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred>0.5))
print(roc_auc_score(y_test, y_pred))

# >> 0.972027972027972



lgbm_params = {
'objective': 'binary',
'device_type': 'cpu',
'deterministic': True,
'random_state': 0,
'verbose': -1,
'verbose_eval': False
}

for i in range(11):
    print(f"{i}, fair param of {(2**i -1)/2**10}")
    clf = FLGBM.FairLGBM(fair_param= (2**i - 1) / 2**10, prot=protname, fair_fun='fpr_diff', lgbm_params=lgbm_params)

    start = time.process_time()
    clf.fit(X_train, y_train, X_val, y_val)
    end = time.process_time()

    y_pred = clf.predict(X_test)
    print("Tiempo LightGBM custom: ", end-start)

    print(accuracy_score(y_test, y_pred))
