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
import warnings
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')


PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/results/'
PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/datasets/data/'

sys.path.insert(1, os.path.abspath(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), 'models')))
from FairDT._classes import DecisionTreeClassifier as FairDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from FLGBM.FLGBM import FairLGBM





def decode(var_range, model, **features):
    """
    Hyperparameter decodificaton and correction: Correct the values of some hyperparameters, as some of them
    can be unbounded when they are generated. Also others convert to integers, and other corrections are performed.
        Parameters:
            - var_range: Range of the hyperparameters, being a list of lists of 2 values (min,max)
            - model: ML model employed
            - features: Hyperparemeter dictionary
        Returns:
            - OrderedDict of the hyperparameters with corrected attributes 
    """
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

        if features['min_samples_split'] is not None:   #If a limit was specified
            features['min_samples_split'] = int(round(features['min_samples_split']))
        else:
            features['max_depth'] = var_range[2][1]

        if features['max_leaf_nodes'] is not None:
            features['max_leaf_nodes'] = int(round(features['max_leaf_nodes']))
        else:
            features['max_leaf_nodes'] = var_range[3][1]

        if features['class_weight'] is not None:        #In case it's None, all classes are supposed to have weight 1
            features['class_weight'] = int(round(features['class_weight']))
        else:
            features['class_weight'] = 5
        
        if features['fair_param'] is None:              #In case it's None, it is supposed that fairness will not be taken into account (classical DT)
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
        if features['num_leaves'] is not None:
            features['num_leaves'] = int(round(features['num_leaves']))
        else:
            features['num_leaves'] = var_range[0][1]
        
        if features['min_data_in_leaf'] is not None:
            features['min_data_in_leaf'] = int(round(features['min_data_in_leaf']))
        else:
            features['min_data_in_leaf'] = var_range[1][1]

        if features['max_depth'] is not None:
            features['max_depth'] = int(round(features['max_depth']))
        else:
            features['max_depth'] = -1
        
        if features['learning_rate'] is None:
            features['learning_rate'] = var_range[3][1]

        if features['n_estimators'] is not None:
            features['n_estimators'] = int(round(features['n_estimators']))
        else:
            features['n_estimators'] = var_range[4][1]
        
        if features['feature_fraction'] is None:
            features['feature_fraction'] = var_range[5][1]

        if features['fair_param'] is None:                          #In case it's None, it is supposed that fairness will not be taken into account (classical DT)
            features['fair_param'] = var_range[6][0]

            
        
        hyperparameters = ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'fair_param']
        
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features





def read_data(df_name):
    """
    Reads the dataset to work with. The base version is read.
        Parameters:
            - df_name: DataFrame name
        Returns:
            - pd.DataFrame containing the specified dataframe.
    """
    df = pd.read_csv(f"{PATH_TO_DATA}{df_name}.csv", sep = ',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    return df





def score_text(v):
    """
    This function will likely be removed
    """
    if v == 'Low':
        return 0
    elif v == 'Medium':
        return 1
    else:
        return 2





def get_matrices(df_name, y_col, seed):
    """
    Data preprocessing and splits dataframe into train and test
        Parameters:
            - df_name: Name of the dataframe to read
            - y_col: Name of the attribute to be predicted, it will be returned separated from the rest
            - seed: Random partition seed
        Returns:
            - X_train, X_val, X_test, y_train, y_val, y_test: Datasets information as pd.Dataframe and series
    """
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





def write_train_val_test(df_name, prot_col, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Exports to csv files train, validation and test sets
        Parameters:
            - df_name: Dataset name
            - prot_col: Protected attribute
            - seed: Random seed partition
            - X_train, X_val, X_test, y_train, y_val, y_test: Dataset information (train, validation and test)
    """
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv(f"{PATH_TO_DATA}train_val_test_standard/{df_name}/{df_name}_{prot_col}_train_seed_{str(seed)}.csv", index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv(f"{PATH_TO_DATA}train_val_test_standard/{df_name}/{df_name}_{prot_col}_val_seed_{str(seed)}.csv", index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv(f"{PATH_TO_DATA}train_val_test_standard/{df_name}/{df_name}_{prot_col}_test_seed_{str(seed)}.csv", index = False)





def print_tree(classifier, model, features):
    """
    Exports obtained decision tree to a png file
        Parameters:
            - classifier: Classifier to export
            - model: ML model used
            - features: Name of the decision tree features
    """
    dot_data = StringIO()
    export_graphviz(classifier, out_file = dot_data, feature_names = features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("../results/" + model + "/trees/tree.png")





def print_properties_tree(learner):
    """
    Returns properties of a Decision Tree
        Parameters:
            - learner: Model to obtain properties
        Returns:
            - depth, leaves, data_avg_depth, depth_unbalance: Calculated DT properties
    """
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    data_avg_depth, depth_unbalance = data_weight_avg_depth(learner)
    return depth, leaves, data_avg_depth, depth_unbalance





def print_properties_lgbm(learner):
    """
    Returns depth and leaves given a decision tree
        Parameters:
            - learner: LGBM model to print properties
        Returns:
            - n_estimators, n_features, feature_importance_std: Calculated LGBM properties
    """
    n_estimators = learner.model.num_trees()
    n_features = learner.model.num_feature()
    feature_importance_std = learner.model.feature_importance('gain').std()
    # Número de árboles real (tree_index)
    #print(learner.model.trees_to_dataframe())
    #print(learner.model.trees_to_dataframe().shape)
    #print(learner.model.trees_to_dataframe().columns)

    return n_estimators, n_features, feature_importance_std





def print_properties_lr(learner):
    """
    Returns coefficients of the given logistic regression
        Parameters:
            - learner: LR model to which calculate its parametesrs
        Returns:
            coefficients
    """
    return learner.coef_





def print_properties_flgbm(learner):
    pass





#TODO Revisar que se ha hecho bien
def train_model(X_train, y_train, prot_col, seed, model, X_val=None, y_val=None, **features):
    """
    Classifier training
        Parameters:
            - X_train: Training dataset
            - y_train: attribute to predict
            - prot_col: Name of the protected attribute
            - seed: Random seed to control reproducibility
            - model: ML model to train
            - features: Other hyperparameters for training
        Return:
            - Trained model    
    """
    prot = X_train[prot_col]

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
                clf = FairDecisionTreeClassifier(criterion = 'gini_fair', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, f_lambda = features['fair_param'], random_state=seed)
            else:
                clf = FairDecisionTreeClassifier(criterion = 'entropy_fair', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, f_lambda = features['fair_param'], random_state=seed)
        else:
            if features['criterion'] <= 0.5:
                clf = FairDecisionTreeClassifier(criterion = 'gini_fair', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], f_lambda = features['fair_param'], random_state=seed)
            else:
                clf = FairDecisionTreeClassifier(criterion = 'entropy_fair', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], f_lambda = features['fair_param'], random_state=seed)
    
    if model == "LR":
        if features['class_weight'] is not None:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=features['max_iter'], tol=features['tol'], C=features['lambda'], l1_ratio=features['l1_ratio'], class_weight = features['class_weight'], random_state=seed)

    if model == "FLGBM":
        lgbm_params = {
        'objective': 'binary',
        'device_type': 'cpu',
        'deterministic': True,
        'random_state': seed,
        'verbose': -1,
        'num_leaves': features['num_leaves'],
        'min_data_in_leaf': features['min_data_in_leaf'],
        'max_depth': features['max_depth'],
        'learning_rate': features['learning_rate'],
        'n_estimators': features['n_estimators'],
        'feature_fraction': features['feature_fraction'],
        'verbose_eval': False
        }
        clf = FairLGBM(fair_param=features['fair_param'], prot=prot_col, fair_fun='fpr_diff', lgbm_params=lgbm_params)
    
    if model == "FDT":
        learner = clf.fit(X_train, y_train, prot = prot.to_numpy())
    elif model == "FLGBM":
        learner = clf.fit(X_train, y_train, X_val, y_val)
    else:
        learner = clf.fit(X_train, y_train, X_val, y_val)

    return learner





def get_max_depth_FLGBM(X_train, y_train, prot_col, seed, **features):
    """
    Auxiliaty function to obtain the max depth of any tree contained in a FLGBM model
    (This will serve to specify parameters range)
        Parameters:
            - X_train: Training dataset
            - y_train: attribute to predict
            - prot_col: Name of the protected attribute
            - seed: Random seed to control reproducibility
            - model: ML model to train
            - features: Other hyperparameters for training
        Return:
            - Max depth found for any tree of the FLGBM 
    """
    lgbm_params = {
    'objective': 'binary',
    'device_type': 'cpu',
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
    
    clf = FairLGBM(fair_param=features['fair_param'], prot=prot_col, fair_fun='fpr_diff', lgbm_params=lgbm_params)
    learner = clf.fit(X_train, y_train)
    return learner.model.trees_to_dataframe()['node_depth'].max()





def save_model(learner, dataset_name, seed, variable_name, num_of_generations, num_of_individuals, individual_id, model, method, objectives):
    """
    Save a given ML model to disk. This function is currently no used
    """
    
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "__" + objectives[i].__name__

    path = PATH_TO_RESULTS + str(model) + '/' + str(method) + '/models/' + dataset_name + "/"
    filename =  'model_ID_' + individual_id + '_seed_' + str(seed) + '_var_' + variable_name + '_gen_' + str(num_of_generations) + '_indiv_' + str(num_of_individuals) + '_obj_' + str(str_obj) + '.sav'
    pickle.dump(learner, open(path + filename, 'wb'))
    return





def val_model(X_val, learner):
    """
    Validates the classifier (comparison with validation set)
        Parameters:
            - X_val: Validation dataset
            - learner: Trained ML classifier
        Returns
            - Results of validation
    """
    return learner.predict(X_val)





def test_model(X_test, learner):
    """
    Validates the classifier (comparison with test set)
        Parameters:
            - X_test: Test dataset
            - learner: Trained ML classifier
        Returns
            - Results of testing
    """
    return learner.predict(X_test)





def evaluate_fairness(X, y, pred, protected_attr, protected_value=1):
    """
    Generates a new dataframe containing protected attribute, attribute to predict and real prediction.
    This dataframe is then divided, deppending on the values of the predicted attribute, and for the 
    real (val) values of y and the predicted ones
        Parameters:
            - X: Complete dataset
            - y: Attribute to predict (real)
            - pred: Prediction made for y
            - protected_attr: Name of the protected attribute
            - protected_value: Value of protected_attr for the privileged group
        Returns:
            - y_val_p: Real y values for privileged class
            - y_val_u: Real y values for unprivileged class
            - y_pred_p: Predicted values for privileged class
            - y_pred_u: Predicted values for unprivileged class
    """
    df = pd.DataFrame({protected_attr: X[protected_attr], 'y_val': y, 'y_pred': pred})
    df_p = df.loc[df[protected_attr] == protected_value, :]        #p variables represent data belonging to privileged class
    df_u = df.loc[df[protected_attr] != protected_value, :]        #u variables represent data belonging to unprivileged class
    y_val_p = df_p['y_val']
    y_val_u = df_u['y_val']
    y_pred_p = df_p['y_pred']
    y_pred_u = df_u['y_pred']
    return y_val_p, y_val_u, y_pred_p, y_pred_u





def accuracy_inv(y_val, y_pred):
    """
    Computes accuracy and does 1 - accuracy (for minimization purposes)
        Parameters:
            - y_val: Real y values,
            - y_pred: Predicted values
        Returns:
            - 1 - Accuracy
    """
    err = 1 - f1_score(y_val, y_pred)
    return err





def gmean_inv(y_val, y_pred):
    """
    Computes Gmean-score (sqrt(TPR*TNR)) and does 1 - accuracy (for minimization purposes)
        Parameters:
            - y_val: Real y values
            - y_pred: Predicted values
        Returns:
            - 1 - Gmean-score
    """
    gmean_error = 1 - geometric_mean_score(y_val, y_pred)
    return gmean_error





def accuracy_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in accuracy rates for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute accuracy difference
    """
    acc_p  = accuracy_score(y_val_p, y_pred_p)
    acc_u = accuracy_score(y_val_u, y_pred_u)
    acc_fair = abs(acc_u - acc_p)
    return acc_fair





def tpr_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in true positive rates for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute TPR difference
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()    #Matriz de confusion de los valores predichos, teniendo en cuenta los verdaderos valores y los predichos
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    
    ret = None
    p_p = tp_p + fn_p
    p_u = tp_u + fn_u

    if(p_p == 0 or p_u == 0):
        ret = 1.0
    else:
        tpr_p = tp_p / p_p
        tpr_u = tp_u / p_u
        ret = abs(tpr_p - tpr_u)
    
    return ret





def fpr_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in false positive rates for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute FPR difference
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()

    ret = None
    n_p = fp_p + tn_p
    n_u = fp_u + tn_u

    if(n_p == 0 or n_u == 0):
        ret = 1.0
    else:
        fpr_p = fp_p / n_p
        fpr_u = fp_u / n_u
        ret = abs(fpr_p - fpr_u)

    return ret





def tnr_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in true negative rates for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute TNR difference
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()

    ret = None
    n_p = fp_p + tn_p
    n_u = fp_u + tn_u

    if(n_p == 0 or n_u == 0):
        ret = 1.0
    else:
        tnr_p = tn_p / n_p
        tnr_u = tn_u / n_u
        ret = abs(tnr_p - tnr_u)

    return ret





def ppv_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in predictive positive values for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute PPV difference
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()

    ret = None
    pp_p = tp_p + fp_p
    pp_u = tp_u + fp_u

    if(pp_p == 0 or pp_u == 0):
        ret = 1.0
    else:
        ppv_p = tp_p / pp_p
        ppv_u = tp_u / pp_u
        ret = abs(ppv_p - ppv_u)

    return ret





def pnr_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Computes difference in predicted negative rates for privileged and unprivileged groups
        Parameters:
            - y_val_p, y_val_u, y_pred_p, y_pred_u: Real y values and protected values for each demographic group
        Returns:
            - Absolute PNR difference
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()

    pnr_p = (fn_p + tn_p)/(tn_p + fp_p + fn_p + tp_p)
    pnr_u = (fn_u + tn_u)/(tn_u + fp_u + fn_u + tp_u)
    dem = abs(pnr_p - pnr_u)
    return dem


def num_leaves(learner):
    """
    First complexity measure. Number of leaves
        Parameters:
            - learner: Trained ML model (DT or FDT)
        Returns:
            - number of leaves
    """
    return int(learner.get_n_leaves())





def data_weight_avg_depth(learner):
    """
    Second complexity measure. Data weighted average depth
        Parameters:
            - learner: Trained ML model (DT or FDT)
        Returns:
            - The value of the metric, as well as the unbalance of the tree
    """
    stack = [(0,0)]                     #Root node id and its depth
    total_w_depth = 0.0
    tree = learner.tree_
    total_samples = 0.0
    min_depth = float('inf')
    max_depth = 0
    while len(stack) > 0:
        current_node, current_depth = stack.pop()
        if(learner.tree_.children_left[current_node] != learner.tree_.children_right[current_node]):    #If it's not a leaf
            stack.append((learner.tree_.children_left[current_node], current_depth + 1))    #Append both children with their depth increased
            stack.append((learner.tree_.children_right[current_node], current_depth + 1))
        else:
            weighted_samples = tree.weighted_n_node_samples[current_node]
            total_w_depth += weighted_samples * current_depth
            total_samples += weighted_samples
            if current_depth < min_depth:
                min_depth = current_depth
            if current_depth > max_depth:
                max_depth = current_depth
    return total_w_depth / total_samples, float(min_depth) / float(max_depth)





def create_generation_stats(model):
    """
    Creates generation stats dataframe, which store data for each generation
    - Parameters:
        - model: learning model
    - Returns:
        - pd.DataFrame of information to store during each generation
    """
    if model == 'FDT' or model == 'DT':
        store_dimensions = ['leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    elif model == 'FLGBM':
        store_dimensions = ['n_estimators', 'n_features', 'feature_importance_std']
    
    new_dict = {}
    for elem in store_dimensions:
        new_dict[f"min_{elem}"] = []
        new_dict[f"mean_{elem}"] = []
        new_dict[f"max_{elem}"] = []
        new_dict[f"std_{elem}"] = []
    new_dict['process_time'] = []
    new_dict['total_time'] = []
    
    return pd.DataFrame(data=new_dict)





def save_generation_stats(generations_df, generation_indivs, model, p_time, t_time):
    """
    Save generation stats of the current generation into the dataframe created using create_genertaion_stats
        Parameters:
            - generations_df: Dataframe containing all previous information
            - generation_indivs: Individuals of the current generation to save stats
            - model: ML model employed
            - p_time: Process time for the current generation to be executed
            - t_time: Total time for the current generation to be executed
        Returns:
            - generations_df with updated information
    """
    if model == 'FDT' or model == 'DT':
        store_dimensions = ['leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    elif model == 'FLGBM':
        store_dimensions = ['n_estimators', 'n_features', 'feature_importance_std']

    new_row = {}
    for elem in store_dimensions:
        new_row[f"min_{elem}"] = generation_indivs[elem].min()
        new_row[f"mean_{elem}"] = generation_indivs[elem].mean()
        new_row[f"max_{elem}"] = generation_indivs[elem].max()
        new_row[f"std_{elem}"] = generation_indivs[elem].std()
    new_row['process_time'] = p_time
    new_row['total_time'] = t_time

    return pd.concat([generations_df, pd.DataFrame([new_row])], ignore_index=True)