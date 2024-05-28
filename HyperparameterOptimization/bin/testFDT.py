import pandas as pd
import numpy as np
import seaborn as sns
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join('..', 'models')))
from FairDT._classes import DecisionTreeClassifier



PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/datasets/data/'
PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/results/GraphicsParamFDT/'
DOTS_TO_PRINT = 100
PRINT_RESULTS = False


# Para estas pruebas voy a trabajar con el conjunto de datos de wisconsin porque es binario y tiene
# mejor pinta que iris para este contexto de clasificación binaria con atributo protegido binario

seeds = []
lambs = []
acc_train = []
acc_test = []
fpr_diff_train = []
fpr_diff_test = []
depths = []
leaves = []
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}


#for dataset in ['adult']:
for dataset in  ['adult', 'compas', 'german', 'ricci', 'obesity', 'insurance', 'student', 'diabetes', 'parkinson', 'dutch']:
    print(f'Calculating values for {dataset} dataset')

    all_exist = (os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_accuracy.pdf') and
                    os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_fpr_diff.pdf') and
                    os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_depth.pdf') and
                    os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_leaves.pdf'))

    if all_exist:
        print('- All results graphics already existed!')
    
    else:
        for seed in range(100, 110):
            train = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_train_seed_{seed}.csv", index_col = False)
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]
            val = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_val_seed_{seed}.csv", index_col = False)
            x_val = val.iloc[:, :-1]
            y_val = val.iloc[:, -1]
            test = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_test_seed_{seed}.csv", index_col = False)
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]



            lamb_test = [0]
            lamb_test += [x/float(DOTS_TO_PRINT) for x in range(1, DOTS_TO_PRINT+1)]

            for lambd in lamb_test:

                clf = DecisionTreeClassifier(random_state=0, criterion="gini_fair", f_lambda=lambd, fair_fun='fpr_diff')
                clf.fit(x_train, y_train, prot=x_train[dict_protected[dataset]].to_numpy())
                y_pred_train = clf.predict(x_train)
                y_pred_test = clf.predict(x_test)

                fpr = []
                for elem in [[y_train, y_pred_train, x_train], [y_test, y_pred_test, x_test]]:
                    fair_df = pd.DataFrame({'y_real': elem[0], 'y_pred': elem[1], 'prot': elem[2][dict_protected[dataset]]} )

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

                seeds.append(seed)
                lambs.append(lambd)
                acc_train.append(1-np.sum(np.abs(y_pred_train - y_train.to_numpy())) / len(y_train))
                acc_test.append(1-np.sum(np.abs(y_pred_test - y_test.to_numpy())) / len(y_test))
                fpr_diff_train.append(fpr[0])
                fpr_diff_test.append(fpr[1])
                depths.append(clf.get_depth())
                leaves.append(clf.get_n_leaves())

                if PRINT_RESULTS:
                    print('-------------------------------')
                    print(f'Lambda: {lambs[-1]}')
                    print(f'Accuracy_train: {acc_train[-1]}')
                    print(f'FPR_diff_train: {fpr_diff_train[-1]}')
                    print('---')
                    print(f'Accuracy_test: {acc_test[-1]}')
                    print(f'FPR_diff_test: {fpr_diff_test[-1]}')
                    print('---')
                    print(f'Profundidad: {depths[-1]}')
                    print(f'Número de hojas: {leaves[-1]}')

        print("- Calculations finished, plotting")

        dict_plot = pd.DataFrame({'Lambda':lambs,
                                    'Acc_train': acc_train,
                                    'FPR_diff_train': fpr_diff_train,
                                    'Acc_val': acc_test,
                                    'FPR_diff_val': fpr_diff_test,
                                    'Depth': depths,
                                    'Leaves': leaves})

        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=dict_plot, x="Lambda", y="Acc_train", ax=plt.gca(),
                    label='Training', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False)

        sns.regplot(data=dict_plot, x="Lambda", y="Acc_val", ax=plt.gca(),
                    label='Test', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
        plt.xlabel('$\lambda$')
        plt.ylabel('Accuracy')
        plt.title(f'Evolution of accuracy for {dataset} dataset using\ndifferent values of the fairness parameter $\lambda$')
        plt.savefig(f'{PATH_TO_RESULTS}{dataset}_accuracy.pdf', format='pdf')

        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=dict_plot, x="Lambda", y="FPR_diff_train", ax=plt.gca(),
                    label='Training', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False, color='#3A8FEB')
        sns.regplot(data=dict_plot, x="Lambda", y="FPR_diff_val", ax=plt.gca(),
                    label='Test', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False, color='#EB473B')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
        plt.xlabel('$\lambda$')
        plt.ylabel('FPR_diff')
        plt.title(f'Evolution of FPR_diff for {dataset} dataset using\ndifferent values of the fairness parameter $\lambda$')
        plt.savefig(f'{PATH_TO_RESULTS}{dataset}_fpr_diff.pdf', format='pdf')

        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=dict_plot, x="Lambda", y="Depth", ax=plt.gca(),
                    label='Depth', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False, color='purple')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
        plt.xlabel('$\lambda$')
        plt.ylabel('Depth')
        plt.title(f'Evolution of depth for {dataset} dataset using\ndifferent values of the fairness parameter $\lambda$')
        plt.savefig(f'{PATH_TO_RESULTS}{dataset}_depth.pdf', format='pdf')

        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=dict_plot, x="Lambda", y="Leaves", ax=plt.gca(),
                    label='Leaves', scatter_kws={'alpha':0.5},
                    x_estimator=np.mean, fit_reg=False, color='green')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
        plt.xlabel('$\lambda$')
        plt.ylabel('Leaves')
        plt.title(f'Evolution of leaves for {dataset} dataset using\ndifferent values of the fairness parameter $\lambda$')
        plt.savefig(f'{PATH_TO_RESULTS}{dataset}_leaves.pdf', format='pdf')

        print('- Execution succesful!')

