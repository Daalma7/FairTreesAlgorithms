import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import time
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import sys
import re
sys.path.insert(1, os.path.abspath(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), 'models')))
from FairDT._classes import DecisionTreeClassifier
from FLGBM.FLGBM import FairLGBM



PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/datasets/data/'
PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/results/TestFLGBM/'
DOTS_TO_PRINT = 100
PRINT_RESULTS = False
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}


fdt_res = None
flgbm_res = None
lambfdt = 0.0
lambflgbm = 0.1

#for dataset in ['adult']:
for dataset in  ['adult', 'compas', 'german', 'ricci', 'obesity', 'insurance', 'student', 'diabetes', 'parkinson', 'dutch']:
    seeds = []
    fdt_gmean_inv_train = []
    fdt_gmean_inv_test = []
    fdt_fpr_diff_train = []
    fdt_fpr_diff_test = []
    fdt_train_time = []
    flgbm_gmean_inv_train = []
    flgbm_gmean_inv_test = []
    flgbm_fpr_diff_train = []
    flgbm_fpr_diff_test = []
    flgbm_train_time = []
    
    
    print(f'- Calculating values for {dataset} dataset')

    all_exist = (os.path.isfile(f"{PATH_TO_RESULTS}{dataset}_FDT_0_fair.csv") and
                    os.path.isfile(f"{PATH_TO_RESULTS}{dataset}_FLGBM_0_fair.csv"))

    if all_exist:
        print('- All results graphics already existed!')
        fdt_res = pd.read_csv(f"{PATH_TO_RESULTS}{dataset}_FDT_0_fair.csv")
        flgbm_res = pd.read_csv(f"{PATH_TO_RESULTS}{dataset}_FLGBM_0_fair.csv")
    else:
        for seed in range(100, 110):

            # Read the data
            train = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_train_seed_{seed}.csv", index_col = False)
            train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]
            val = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_val_seed_{seed}.csv", index_col = False)
            val = val.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            x_val = val.iloc[:, :-1]
            y_val = val.iloc[:, -1]
            test = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_test_seed_{seed}.csv", index_col = False)
            test = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]
            lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)

            print(x_train.columns)


            # Specify parameters
            np.random.seed(7)
            lgbm_params = {
            'objective': 'binary',
            'device_type': 'cpu',
            'deterministic': True,
            'random_state': 0,
            'verbose': -1,
            'verbose_eval': False,
            'num_threads': 16
            }

            seeds.append(seed)
            print('Seed ', seed)
            
            # Train
            for model in ['FDT', 'FLGBM']:

                clf = None
                cut_time = 0
                if model == 'FDT':
                    print(f"- Training FDT with lambda = {lambfdt}")
                    start = time.process_time()
                    clf = DecisionTreeClassifier(random_state=0, criterion="gini_fair", f_lambda=0, fair_fun='fpr_diff')
                    clf.fit(x_train, y_train, prot=x_train[dict_protected[dataset]].to_numpy())
                    end = time.process_time()
                    curtime = end-start
                elif model == 'FLGBM':
                    print(f"- Train FLGBM with lambda = {lambflgbm}")
                    start = time.process_time()
                    clf = FairLGBM(fair_param=0.1, prot=dict_protected[dataset], fair_fun='fpr_diff', lgbm_params=lgbm_params)
                    clf.fit(x_train, y_train, x_val, y_val)
                    end = time.process_time()
                    curtime = end-start

                    print(f"- Tiempo FLGBM normal: {end-start}")

                y_pred_train = clf.predict(x_train)
                y_pred_test = clf.predict(x_test)

                gmeans = []
                fpr = []
                for elem in [[y_train, y_pred_train, x_train], [y_test, y_pred_test, x_test]]:
                    fair_df = pd.DataFrame({'y_real': elem[0], 'y_pred': elem[1], 'prot': elem[2][dict_protected[dataset]]} )

                    # GMEAN_INV calculation
                    tp = fair_df.loc[(fair_df['y_real'] == 1) & (fair_df['y_pred'] == 1), :].shape[0]
                    p = fair_df.loc[fair_df['y_real'] == 1, :].shape[0]
                    tn = fair_df.loc[(fair_df['y_real'] == 0) & (fair_df['y_pred'] == 0), :].shape[0]
                    n = fair_df.loc[fair_df['y_real'] == 0, :].shape[0]

                    gmeans.append(1 - np.sqrt((tp/p) * (tn/n)))


                    priv = fair_df.loc[fair_df['prot'] == 1, :]
                    unpriv = fair_df.loc[fair_df['prot'] == 0, :]

                    tp_p = priv.loc[(priv['y_real'] == 1) & (priv['y_pred'] == 1)].shape[0]
                    fp_p = priv.loc[(priv['y_real'] == 0) & (priv['y_pred'] == 1)].shape[0]
                    tn_p = priv.loc[(priv['y_real'] == 0) & (priv['y_pred'] == 0)].shape[0]
                    fn_p = priv.loc[(priv['y_real'] == 1) & (priv['y_pred'] == 0)].shape[0]

                    tp_u = unpriv.loc[(unpriv['y_real'] == 1) & (unpriv['y_pred'] == 1)].shape[0]
                    fp_u = unpriv.loc[(unpriv['y_real'] == 0) & (unpriv['y_pred'] == 1)].shape[0]
                    tn_u = unpriv.loc[(unpriv['y_real'] == 0) & (unpriv['y_pred'] == 0)].shape[0]
                    fn_u = unpriv.loc[(unpriv['y_real'] == 1) & (unpriv['y_pred'] == 0)].shape[0]

                    fpr_diff = fpr_p = fpr_u = 0
                    n_p = fp_p + tn_p
                    n_u = fp_u + tn_u

                    if(n_p == 0 or n_u == 0):
                        fpr_diff = 1.0
                    else:
                        fpr_p = fp_p / n_p
                        fpr_u = fp_u / n_u
                        fpr_diff = abs(fpr_p - fpr_u)
                    
                    fpr.append(fpr_diff)

                if model == 'FDT':
                    fdt_gmean_inv_train.append(gmeans[0])
                    fdt_gmean_inv_test.append(gmeans[1])
                    fdt_fpr_diff_train.append(fpr[0])
                    fdt_fpr_diff_test.append(fpr[1])
                    fdt_train_time.append(curtime)
                elif model == 'FLGBM':
                    flgbm_gmean_inv_train.append(gmeans[0])
                    flgbm_gmean_inv_test.append(gmeans[1])
                    flgbm_fpr_diff_train.append(fpr[0])
                    flgbm_fpr_diff_test.append(fpr[1])
                    flgbm_train_time.append(curtime)

        fdt_res = pd.DataFrame({'Seed': seeds, 'Gmean_inv_train': fdt_gmean_inv_train, 'FPR_train': fdt_fpr_diff_train, 'Gmean_inv_test': fdt_gmean_inv_test, 'FPR_test': fdt_fpr_diff_test, 'Training_time': fdt_train_time})
        flgbm_res = pd.DataFrame({'Seed': seeds, 'Gmean_inv_train': flgbm_gmean_inv_train, 'FPR_train': flgbm_fpr_diff_train, 'Gmean_inv_test': flgbm_gmean_inv_test, 'FPR_test': flgbm_fpr_diff_test, 'Training_time': flgbm_train_time})



    print(f"Total results for dataset: {dataset}")
    print('---- FDT ----')
    print(fdt_res)
    print('--- FLGBM ---')
    print(flgbm_res)

    
    print(f"Mean results for dataset: {dataset}")
    print('---- FDT ----')
    print(f"- Gmean_inv_train: {fdt_res['Gmean_inv_train'].mean()}")
    print(f"- FPR_train: {fdt_res['FPR_train'].mean()}")
    print(f"- Gmean_inv_test: {fdt_res['Gmean_inv_test'].mean()}")
    print(f"- FPR_test: {fdt_res['FPR_test'].mean()}") 
    print(f"- Training_time: {fdt_res['Training_time'].mean()}")
    print('--- FLGBM ---')
    print(f"- Gmean_inv_train: {flgbm_res['Gmean_inv_train'].mean()}")
    print(f"- FPR_train: {flgbm_res['FPR_train'].mean()}")
    print(f"- Gmean_inv_test: {flgbm_res['Gmean_inv_test'].mean()}")
    print(f"- FPR_test: {flgbm_res['FPR_test'].mean()}") 
    print(f"- Training_time: {flgbm_res['Training_time'].mean()}")

    fdt_res.to_csv(f"{PATH_TO_RESULTS}{dataset}_FDT_0_fair.csv", index=False)        
    flgbm_res.to_csv(f"{PATH_TO_RESULTS}{dataset}_FLGBM_0_fair.csv", index=False)        