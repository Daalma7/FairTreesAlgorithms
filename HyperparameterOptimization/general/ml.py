import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import sys, os
from math import ceil
import collections
from six import StringIO
import pydotplus
from imblearn.metrics import geometric_mean_score
from collections import Counter
import contextlib
import os
import re
import pickle


PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/results/'
PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/datasets/data/'

sys.path.insert(1, os.path.abspath(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), 'models')))
from FairDT._classes import DecisionTreeClassifier as FairDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from FLGBM.FLGBM import FairLGBM


#Decoding hyperparameters
# Here we correct the values of some hyperparameters, as some of them can be unbounded
# and when they generate, they are all continuous, and some need to be converted to integers.
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
    
    if model == "FDT":
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
        
        if features['fair_param'] is not None:
            features['fair_param'] = int(round(features['fair_param']))
        else:                                           #In case it's None, it is supposed that fairness will not be taken into account (classical DT)
            features['fair_param'] = var_range[5][0]

        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']

    if model == "LR":
        if features['max_iter'] is not None:
            features['max_iter'] = int(round(features['max_iter']))
        else:
            features['max_iter'] = var_range[0][1]
        
        if features['class_weight'] is not None:
            features['class_weight'] = int(round(features['class_weight']))
        #Other parameters are float, so they haven't got to be further processed
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        
    if model == "FLGBM":
        if features['lamb'] is None:
            features['lamb'] = var_range[0][1]
        
        if features['num_leaves'] is not None:
            features['num_leaves'] = int(round(features['num_leaves']))
        else:
            features['num_leaves'] = var_range[1][1]
        
        if features['min_data_in_leaf'] is not None:
            features['min_data_in_leaf'] = int(round(features['min_data_in_leaf']))
        else:
            features['min_data_in_leaf'] = var_range[2][1]

        if features['max_depth'] is not None:
            features['max_depth'] = int(round(features['max_depth']))
        else:
            features['max_depth'] = -1
        
        if features['learning_rate'] is None:
            features['learning_rate'] = var_range[4][1]

        if features['n_estimators'] is not None:
            features['n_estimators'] = int(round(features['n_estimators']))
        else:
            features['n_estimators'] = var_range[5][1]
        
        if features['feature_fraction'] is None:
            features['feature_fraction'] = var_range[6][1]
            
        
        hyperparameters = ['lamb', 'num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction']
        
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features

#Reads the dataset to work with. You have to work with preprocessed data, and to ensure it, we will only read the files ending with _preproc
def read_data(df_name):
    df = pd.read_csv(PATH_TO_DATA + df_name + '.csv', sep = ',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    return df

def score_text(v):
    if v == 'Low':
        return 0
    elif v == 'Medium':
        return 1
    else:
        return 2


#Data preprocessing and splits dataframe into train and test
def get_matrices(df_name, y_col, seed):
    df = read_data(df_name)

    if df_name == 'compas':
        df = df.drop('decile_score', axis=1)
    else:
        df = df.drop(y_col, axis=1)
    if y_col + '_binary' in df.columns:
        df[y_col] = df[y_col + '_binary']
        df = df.drop(y_col + '_binary', axis=1)
    if 'binary_' + y_col in df.columns:
        df[y_col] = df['binary_' + y_col]
        df = df.drop('binary_' + y_col, axis=1)
    
    X = df.loc[:, df.columns != y_col]
    y = df.loc[:, y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

#Exports to csv files train, validation and test sets
def write_train_val_test(df_name, prot_col, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + prot_col + '_train_seed_' + str(seed) + '.csv', index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + prot_col + '_val_seed_' + str(seed) + '.csv', index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + prot_col + '_test_seed_' + str(seed) + '.csv', index = False)

#Exports obtained decision tree to a png file
def print_tree(classifier, model, features):
    dot_data = StringIO()
    export_graphviz(classifier, out_file = dot_data, feature_names = features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("../results/" + model + "/trees/tree.png")

#Returns depth and leaves given a decision tree
def print_properties_tree(learner):
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    w_avg_depth = data_weight_avg_depth(learner)
    return depth, leaves, w_avg_depth

#Returns coefficients of the given logistic regression
def print_properties_lr(learner):
    return learner.coef_

def print_properties_flgbm(learner):
    pass

#TODO Revisar que se ha hecho bien
#Classifier training
def train_model(df_name, variable, seed, model, **features):
    #path = PATH_TO_DATA + 'train_val_test_standard/' + df_name
    #pattern = r'.*binary_train_seed_%s\.csv' % seed
    #print(os.path.abspath(__file__))
    #matching_files = [file for file in os.listdir(path) if re.match(pattern, file)]
    #pd.read_csv(path + '/' + matching_files[0])
    train = pd.read_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + variable + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    prot = X_train[variable]
    
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
    
    if model == "FDT":
        if features['class_weight'] is not None:
            if(features['criterion'] <= 0.5):
                clf = FairDecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, f_lambda = float(features['fair_param'] / 10.0), random_state=seed)
            else:
                clf = FairDecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, f_lambda = float(features['fair_param'] / 10.0), random_state=seed)
        else:
            if features['criterion'] <= 0.5:
                clf = FairDecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], f_lambda = float(features['fair_param'] / 10.0), random_state=seed)
            else:
                clf = FairDecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], f_lambda = float(features['fair_param'] / 10.0), random_state=seed)
    
    if model == "LR":
        if features['class_weight'] is not None:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = features['class_weight'], random_state=seed)

    if model == "FLGBM":
        lgbm_params = {
        'objective': 'binary',
        'deterministic': True,
        'random_state': seed,
        'verbose': -1,
        'num_leaves': features['num_leaves'],
        'min_data_in_leaf': features['min_data_in_leaf'],
        'max_depth': features['max_depth'],
        'learning_rate': features['learning_rate'],
        'n_estimators': features['n_estimators'],
        'feature_fraction': features['feature_fraction']
        }
        clf = FairLGBM(lamb=features['lamb'], proc=variable, fair_c='fpr', lgbm_params=lgbm_params)
    
    if model == "FDT":
        learner = clf.fit(X_train, y_train, prot = prot.to_numpy())
    else:
        learner = clf.fit(X_train, y_train)

    return learner


# TODO: hola
def get_max_depth_FLGBM(df_name, variable, seed, model, **features):
    train = pd.read_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + variable + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    prot = X_train[variable]
    
    lgbm_params = {
    'objective': 'binary',
    'deterministic': True,
    'random_state': seed,
    'verbose': 2,
    'num_leaves': features['num_leaves'],
    'min_data_in_leaf': features['min_data_in_leaf'],
    'max_depth': features['max_depth'],
    'learning_rate': features['learning_rate'],
    'n_estimators': features['n_estimators'],
    'feature_fraction': features['feature_fraction']
    }
    
    clf = FairLGBM(lamb=features['lamb'], proc=variable, fair_c='fpr', lgbm_params=lgbm_params)
    
    with open('../data/output.txt', 'w') as f, contextlib.redirect_stdout(f):
        learner = clf.fit(X_train, y_train)

    depths = []
    with open('../data/output.txt', 'r') as f:
        for line in f.readlines():
            if len(line.split()) > 2 and line.split()[-3] == 'depth':
                depths.append(int(line.split()[-1]))
        
    return max(depths)

def save_model(learner, dataset_name, seed, variable_name, num_of_generations, num_of_individuals, individual_id, model, method, objectives):
    # save the model to disk
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "__" + objectives[i].__name__

    path = PATH_TO_RESULTS + str(model) + '/' + str(method) + '/models/' + dataset_name + "/"
    filename =  'model_id_' + individual_id + '_seed_' + str(seed) + '_var_' + variable_name + '_gen_' + str(num_of_generations) + '_indiv_' + str(num_of_individuals) + '_obj_' + str(str_obj) + '.sav'
    pickle.dump(learner, open(path + filename, 'wb'))
    return

#TODO Revisar que se ha hecho bien
#Validates the classifier (comparison with validation set)
def val_model(df_name, variable, learner, seed):
    val = pd.read_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + variable + '_val_seed_' + str(seed) + '.csv')
    X_val = val.iloc[:, :-1]
    y_val = val.iloc[:, -1]
    y_pred = learner.predict(X_val)
    return X_val, y_val, y_pred

#TODO Revisar que se ha hecho bien
#Tests the classifier (comparison with test set)
def test_model(df_name, variable, learner, seed):
    test = pd.read_csv(PATH_TO_DATA + 'train_val_test_standard/' + df_name + '/' + df_name + '_' + variable + '_test_seed_' + str(seed) + '.csv')
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    y_pred = learner.predict(X_test)
    return X_test, y_test, y_pred

#Split dataset using protected attribute
    #y_val_p values belonging to privileged class
def split_protected(X, y, pred, protected_variable, protected_value = 1):
    df = pd.DataFrame({protected_variable: X[protected_variable], 'y_val': y, 'y_pred': pred})
    df_p = df.loc[df[protected_variable] == protected_value, :]        #p variables represent data belonging to privileged class
    df_u = df.loc[df[protected_variable] != protected_value, :]        #u variables represent data belonging to unprivileged class
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
def data_weight_avg_depth(learner):
    stack = [(0,0)]                     #Root node id and its depth
    total_w_depth = 0.0
    tree = learner.tree_
    total_samples = 0.0
    while len(stack) > 0:
        current_node, current_depth = stack.pop()
        if(learner.tree_.children_left[current_node] != learner.tree_.children_right[current_node]):    #If it's not a leaf
            stack.append((learner.tree_.children_left[current_node], current_depth + 1))    #Append both children with their depth increased
            stack.append((learner.tree_.children_right[current_node], current_depth + 1))
        else:
            weighted_samples = tree.weighted_n_node_samples[current_node]
            total_w_depth += weighted_samples * current_depth
            total_samples += weighted_samples
    return total_w_depth / total_samples


def create_generation_stats(model):
    if model == 'FDT' or model == 'DT':
        return pd.DataFrame(data={'min_depth':[], 'mean_depth':[], 'max_depth':[], 'std_depth':[], 'min_leaves': [], 'mean_leaves': [], 'max_leaves':[], 'std_leaves': [], 'min_data_avg_depth': [], 'mean_data_avg_depth':[], 'max_data_avg_depth':[], 'std_data_avg_depth':[], 'process_time':[]})
    elif model == 'FLGBM':
        return pd.DataFrame(data={})

def save_generation_stats(generations_df, generation_indivs, model, newtime):
    if model == 'FDT' or model == 'DT':
        new_row ={
        'min_depth':generation_indivs['actual_depth'].min(),
        'mean_depth':generation_indivs['actual_depth'].mean(),
        'max_depth':generation_indivs['actual_depth'].max(),
        'std_depth':generation_indivs['actual_depth'].std(),

        'min_leaves':generation_indivs['actual_leaves'].min(),
        'mean_leaves':generation_indivs['actual_leaves'].mean(),
        'max_leaves':generation_indivs['actual_leaves'].max(),
        'std_leaves':generation_indivs['actual_leaves'].std(),

        'min_data_avg_depth':generation_indivs['actual_data_avg_depth'].min(),
        'mean_data_avg_depth':generation_indivs['actual_data_avg_depth'].mean(),
        'max_data_avg_depth':generation_indivs['actual_data_avg_depth'].max(),
        'std_data_avg_depth':generation_indivs['actual_data_avg_depth'].std(),

        'process_time': newtime
        }

        return generations_df.append(new_row, ignore_index=True)

    # TODO:Completar bien
    elif model== 'FLGBM':
        new_row ={
        'min_n_estimators':generation_indivs['n_estimators'].min(),
        'mean_n_estimators':generation_indivs['n_estimators'].mean(),
        'min_n_estimators':generation_indivs['n_estimators'].max(),
        'min_n_estimators':generation_indivs['n_estimators'].std(),

        'min_n_features':generation_indivs['n_features'].min(),
        'mean_n_features':generation_indivs['n_features'].mean(),
        'max_n_features':generation_indivs['n_features'].max(),
        'std_n_features':generation_indivs['n_features'].std(),

        'min_feature_importance_std':generation_indivs['feature_importance_std'].min(),
        'mean_feature_importance_std':generation_indivs['feature_importance_std'].mean(),
        'max_feature_importance_std':generation_indivs['feature_importance_std'].max(),
        'std_feature_importance_std':generation_indivs['feature_importance_std'].std(),

        'process_time': newtime
        }

        return generations_df.append(new_row, ignore_index=True)
