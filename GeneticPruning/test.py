from sklearn import tree
import numpy as np
import pandas as pd
import graphviz
from sklearn.metrics import accuracy_score


from genetic import Genetic_Pruning_Process_NSGA2
from individual import Tree_Structure

for seed in range(100, 101):
    train = pd.read_csv('../data/train_val_test/adult_train_seed_' + str(seed) + '.csv')
    val = pd.read_csv('../data/train_val_test/adult_val_seed_' + str(seed) + '.csv')
    test = pd.read_csv('../data/train_val_test/adult_test_seed_' + str(seed) + '.csv')
    
    y_train = train['y']
    x_train = train.loc[:, train.columns!='y']
    prot_train = train['race']
    
    y_val = val['y']
    x_val = val.loc[:, val.columns!='y']
    prot_val = val['race']

    y_test = test['y']
    x_test = test.loc[:, test.columns!='y']
    prot_test = test['race']

    
    prot = train['race']

    
    struc = Tree_Structure(x_train, y_train, prot_train, x_val, y_val, prot_val)
    gen_process = Genetic_Pruning_Process_NSGA2(struc, ['accuracy', 'fpr_diff'], 10, 50, 0.7, 0.2)
    indivs = gen_process.genetic_optimization(777)
    trees = []
    for indiv in indivs:
        print("-------------")
        trees.append(indiv.get_tree())
        print(accuracy_score(y_test, trees[-1].predict(x_test)))

        

# Use validation and test dataframes:
