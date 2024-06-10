from sklearn import tree
import numpy as np
import pandas as pd
import graphviz
import os
from sklearn.metrics import accuracy_score


from genetic import Genetic_Pruning_Process_NSGA2
from individual import Tree_Structure



PATH_TO_DATA = os.path.dirname(os.path.dirname(__file__)) + '/datasets/data/'
PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/results/TestFGP/'
DOTS_TO_PRINT = 100
PRINT_RESULTS = False
CHECK_STRUC = False
TEST_EXEC = True


dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}


if CHECK_STRUC:
    last_struc = None
    # Read the data
    for dataset in ['diabetes']:

        for i in range(10):
            print(i)
            train = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_train_seed_100.csv", index_col = False)
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]
            val = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_val_seed_100.csv", index_col = False)
            x_val = val.iloc[:, :-1]
            y_val = val.iloc[:, -1]
            test = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_test_seed_100.csv", index_col = False)
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]

            struc = Tree_Structure(x_train,
                                    y_train,
                                    x_train[dict_protected[dataset]].astype(int),
                                    x_val,
                                    y_val,
                                    x_val[dict_protected[dataset]].astype(int),
                                    combine_train_val=False)
        
            if last_struc is None:
                print("Inicializando")
                last_struc = struc
            else:
                if (last_struc.pruning_space == struc.pruning_space) and\
                    (last_struc.fair_dict == last_struc.fair_dict) and\
                    (last_struc.total_samples_dict == last_struc.total_samples_dict) and\
                    (last_struc.base_leaves == last_struc.base_leaves) and\
                    (last_struc.assoc_dict == last_struc.assoc_dict):
                    print('Ambos son iguales')
                else:
                    print('NO SON IGUALES')
                last_struc = struc



if TEST_EXEC:

    seeds = []
    lambs = []
    acc_train = []
    acc_test = []
    fpr_diff_train = []
    fpr_diff_test = []
    depths = []
    leaves = []


    for dataset in ['diabetes']:
    #for dataset in  ['adult', 'compas', 'german', 'ricci', 'obesity', 'insurance', 'student', 'diabetes', 'parkinson', 'dutch']:
        print(f'Calculating values for {dataset} dataset')

        """
        all_exist = (os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_accuracy.pdf') and
                        os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_fpr_diff.pdf') and
                        os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_depth.pdf') and
                        os.path.isfile(f'{PATH_TO_RESULTS}{dataset}_leaves.pdf'))

        if all_exist:
            print('- All results graphics already existed!')
        
        else:
        """
        for seed in range(100, 101):

            # Read the data
            train = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_train_seed_{seed}.csv", index_col = False)
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]
            val = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_val_seed_{seed}.csv", index_col = False)
            x_val = val.iloc[:, :-1]
            y_val = val.iloc[:, -1]
            test = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{dataset}/{dataset}_{dict_protected[dataset]}_test_seed_{seed}.csv", index_col = False)
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]
            
            # Define the algorithm's environment
            struc = Tree_Structure(x_train,
                                y_train,
                                x_train[dict_protected[dataset]].astype(int),
                                x_val,
                                y_val,
                                x_val[dict_protected[dataset]].astype(int),
                                combine_train_val=False)
            gen_process = Genetic_Pruning_Process_NSGA2(struc,
                                                        objs = ['gmean_inv', 'fpr_diff'],
                                                        num_gen = 20,
                                                        num_indiv = 150,
                                                        prob_cross = 1.0,
                                                        prob_mutation = 0.5)
            
            # Execute the algorithm and get all decision trees found
            indivs, evo_stats_df, pop_stats_df = gen_process.genetic_optimization(777, parallel=False)

            repre = []
            gmean_inv_train = []
            fpr_diff_train = []
            gmean_inv_val = []
            fpr_diff_val = []
            gmean_inv_test = []
            fpr_diff_test = []

            for indiv in indivs:

                objs_val, objs_train = indiv.calc_objectives()
                repre.append(indiv.repre)
                gmean_inv_val.append(objs_val[0])
                fpr_diff_val.append(objs_val[1])
                gmean_inv_train.append(objs_train[0])
                fpr_diff_train.append(objs_train[1])

                y_pred_test = indiv.get_tree().predict(x_test)            
                fair_df = pd.DataFrame({'y_real': y_test, 'y_pred': y_pred_test, 'prot': x_test[dict_protected[dataset]]} )

                # GMEAN_INV calculation
                tp = fair_df.loc[(fair_df['y_real'] == 1) & (fair_df['y_pred'] == 1), :].shape[0]
                p = fair_df.loc[fair_df['y_real'] == 1, :].shape[0]
                tn = fair_df.loc[(fair_df['y_real'] == 0) & (fair_df['y_pred'] == 0), :].shape[0]
                n = fair_df.loc[fair_df['y_real'] == 0, :].shape[0]

                gmean_inv_test.append(1 - np.sqrt((tp/p) * (tn/n)))


                #FPR_DIFF calculation
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
                
                fpr_diff_test.append(fpr_diff)

            
            print_df = pd.DataFrame({'gmean_inv_train': gmean_inv_train, 'fpr_diff_train': fpr_diff_train,
                                'gmean_inv_val': gmean_inv_val, 'fpr_diff_val': fpr_diff_val,
                                'gmean_inv_test': gmean_inv_test, 'fpr_diff_test': fpr_diff_test,
                                'repre': repre})

            print(print_df)
            
            print_df.to_csv(f"{PATH_TO_RESULTS}/results_test.csv")

    
        # COSAS PARA HACER:
            #- 1: GUARDAR LOS RESULTADOS EN UN CSV
            #- 2: HACER QUE NO PUEDA HABER INDIVIDUOS CON LAS MISMAS REPRESENTACIONES EN UNA POBLACIÓN

        # Use validation and test dataframes:
