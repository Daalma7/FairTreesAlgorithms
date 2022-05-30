import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import yaml
from math import ceil
import collections
from six import StringIO
from IPython.display import Image
import pydotplus
from imblearn.metrics import geometric_mean_score
from collections import Counter

import pickle

#Decoding hyperparameters
def decode(var_range, model, **features):

    if model == "DT":
        features['criterion'] = int(round(features['criterion'], 0))

        if features['max_depth'] is not None:   #If a limit was specified
            features['max_depth'] = int(round(features['max_depth']))
        else:                                   #In other case, we use the upper predefined bound (in fairness.py)
            features['max_depth'] = var_range[1][1]

        features['min_samples_split'] = int(round(features['min_samples_split']))

        if features['max_leaf_nodes'] is not None:
            features['max_leaf_nodes'] = int(round(features['max_leaf_nodes']))
        else:
            features['max_leaf_nodes'] = var_range[3][1]

        if features['class_weight'] is not None:        #In case it's None, all classes are supposed to have weight 1
            features['class_weight'] = int(round(features['class_weight']))

        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']

    if model == "LR":
        if features['max_iter'] is not None:
            features['max_iter'] = int(round(features['max_iter']))
        else:
            features['max_iter'] = var_range[0][1]
        
        if features['class_weight'] is not None:
            features['class_weight'] = int(round(features['class_weight']))
        #Other parameters are float, so they haven't got to be further processed
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features

#Reads the dataset to work with. You have to work with preprocessed data, and to ensure it, we will only read the files ending with _preproc
def read_data(df_name):
        
    df = pd.read_csv('../data/' + df_name + '_preproc.csv', sep = ',')
    return df

def score_text(v):
    if v == 'Low':
        return 0
    elif v == 'Medium':
        return 1
    else:
        return 2


#Data preprocessing and splits dataframe into train and test
def get_matrices(df_name, seed):
    df = read_data(df_name)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

#Exports to csv files train, validation and test sets
def write_train_val_test(df_name, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv('../data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv', index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv('../data/train_val_test/' + df_name + '_val_seed_' + str(seed) + '.csv', index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv('../data/train_val_test/' + df_name + '_test_seed_' + str(seed) + '.csv', index = False)

#Exports obtained decision tree to a png file
def print_tree(classifier, features):
    dot_data = StringIO()
    export_graphviz(classifier, out_file = dot_data, feature_names = features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("../results/trees/tree.png")

#Returns depth and leaves given a decision tree
def print_properties_tree(learner):
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    return depth, leaves

#Returns coefficients of the given logistic regression
def print_properties_lr(learner):
    return learner.coef_

#Classifier training
def train_model(df_name, seed, model, **features):
    train = pd.read_csv('../data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    #We will need to use the seed used to split into train and test also as seed for these methods, because as they are trained twice, we have to be exactly the same both times
    if model == "DT":
        if features['class_weight'] is not None:
            if(features['criterion'] <= 0.5):
                clf = DecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, random_state=seed)
            else:
                clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, random_state=seed)
        else:
            if features['criterion'] <= 0.5:
                clf = DecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], random_state=seed)
            else:
                clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], random_state=seed)
    
    if model == "LR":
        if features['class_weight'] is not None:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = features['class_weight'], random_state=seed)


    learner = clf.fit(X_train, y_train)
    return learner

def save_model(learner, dataset_name, seed, variable_name, num_of_generations, num_of_individuals, individual_id, model, method, objectives):
    # save the model to disk
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "__" + objectives[i].__name__

    path = '../results/' + str(method) + '/models/' + model + '/' + dataset_name + "/"
    filename =  'model_id_' + individual_id + '_seed_' + str(seed) + '_var_' + variable_name + '_gen_' + str(num_of_generations) + '_indiv_' + str(num_of_individuals) + '_obj_' + str(str_obj) + '.sav'
    pickle.dump(learner, open(path + filename, 'wb'))
    return

#Validates the classifier (comparison with validation set)
def val_model(df_name, learner, seed):
    val = pd.read_csv('../data/train_val_test/' + df_name + '_val_seed_' + str(seed) + '.csv')
    X_val = val.iloc[:, :-1]
    y_val = val.iloc[:, -1]
    y_pred = learner.predict(X_val)
    return X_val, y_val, y_pred

#Tests the classifier (comparison with test set)
def test_model(df_name, learner, seed):
    test = pd.read_csv('../data/train_val_test/' + df_name + '_test_seed_' + str(seed) + '.csv')
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    y_pred = learner.predict(X_test)
    return X_test, y_test, y_pred

#Split dataset using protected attribute
    #y_val_p values belonging to privileged class
def split_protected(X, y, pred, protected_variable, protected_value = 1):
    df = pd.DataFrame({protected_variable: X[protected_variable], 'y_val': y, 'y_pred': pred})
    df_p = df.loc[df[protected_variable] == protected_value]        #p variables represent data belonging to privileged class
    df_u = df.loc[df[protected_variable] != protected_value]        #u variables represent data belonging to unprivileged class
    y_val_p = df_p['y_val']
    y_val_u = df_u['y_val']
    y_pred_p = df_p['y_pred']
    y_pred_u = df_u['y_pred']
    return y_val_p, y_val_u, y_pred_p, y_pred_u

#La evaluación del fairness consiste en una llamada a split_protected
def evaluate_fairness(X_val, y_val, y_pred, protected_variable):
    y_val_p, y_val_u, y_pred_p, y_pred_u = split_protected(X_val, y_val, y_pred, protected_variable, 1)
    return y_val_p, y_val_u, y_pred_p, y_pred_u

#Classical quality measure
def accuracy_inv(y_val, y_pred):
    err = 1 - f1_score(y_val, y_pred)
    return err

#Quality measure
def gmean_inv(y_val, y_pred):
    gmean_error = 1 - geometric_mean_score(y_val, y_pred)
    return gmean_error

#Difference of accuracies
def accuracy_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    acc_p  = accuracy_score(y_val_p, y_pred_p)
    acc_u = accuracy_score(y_val_u, y_pred_u)
    acc_fair = abs(acc_u - acc_p)
    return acc_fair


#TPR: True Positive Rate: Verdaderos positivos entre todos los positivos (verdaderos positivos y falsos negativos)
def dem_tpr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute demography metric.
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()    #Matriz de confusion de los valores predichos, teniendo en cuenta los verdaderos valores y los predichos
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tpr_p = tp_p/(tp_p + fn_p)
    tpr_u = tp_u/(tp_u + fn_u)
    dem = abs(tpr_p - tpr_u)
    if(tpr_p == 0 or tpr_u == 0):
        dem = 1
    return dem

#FPR: False Positive Rate: Falsos positivos entre todos los negativos (falsos positivos y verdaderos negativos)
def dem_fpr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute false positive rate parity.
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    fpr_p = fp_p/(fp_p + tn_p)
    fpr_u = fp_u/(fp_u + tn_u)
    dem = abs(fpr_p - fpr_u)
    if(fpr_p == 0 or fpr_u == 0):
        dem = 1
    return dem

#TNR: True Negative Rate: Verdaderos negativos entre todos los negativos (verdaderos negativos y falsos positivos)
def dem_tnr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tnr_p = tn_p/(tn_p + fp_p)
    tnr_u = tn_u/(tn_u + fp_u)
    dem = abs(tnr_p - tnr_u)
    return dem

#PPV: Positive Predictive Value: Verdaderos positivos entre los predichos como positivos (verdaderos positivos y falsos positivos)
def dem_ppv(y_val_p, y_val_u, y_pred_p, y_pred_u):
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    ppv_p = tp_p/(tp_p + fp_p)
    ppv_u = tp_u/(tp_u + fp_u)
    dem = abs(ppv_p - ppv_u)
    if(tp_p == 0 or tp_u ==0):
        dem = 1
    return dem

#PNR: Predicted Negative Rate: Predichos como negativos entre todos los valores (MEDIDA PARA DEMOGRAPHIC PARITY)
def dem_pnr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    pnr_p = (fn_p + tn_p)/(tn_p + fp_p + fn_p + tp_p)
    pnr_u = (fn_u + tn_u)/(tn_u + fp_u + fn_u + tp_u)
    dem = abs(pnr_p - pnr_u)
    
    return dem

#First complexity measure.
#Returns the number of leaves that the learner has (divided by div which is expected to be the maximum number of leaves that can be got)
def num_leaves(learner):
    return int(learner.get_n_leaves())


#Second complexity measure, that complements the first one.
#Return the weighted average of the depth of all leaves nodes, considering the number of training samples that fell on each one.
def data_weight_avg_depth(learner, data, seed):
    leaves_index = learner.apply(data)          #We get the leaf indices where data examples ended.
    cnt = Counter(leaves_index)                 #We count the number of element of each leaf
    num_data = len(data.index)              #Total number of train examples
    stack = [(0,0)]                     #Root node id and its depth
    total_depth = 0.0
    while len(stack) > 0:
        current_node, current_depth = stack.pop()
        if(learner.tree_.children_left[current_node] != learner.tree_.children_right[current_node]):    #If it's not a leaf
            stack.append((learner.tree_.children_left[current_node], current_depth + 1))    #Append both children with their depth increased
            stack.append((learner.tree_.children_right[current_node], current_depth + 1))
        else:
            total_depth += current_depth * cnt[current_node] / float(num_data)
    return total_depth