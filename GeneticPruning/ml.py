import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from six import StringIO
import pydotplus
from collections import Counter
import string
import random

import pickle


#Reads the dataset to work with. You have to work with preprocessed data, and to ensure it, we will only read the files ending with _preproc
def read_data(df_name):
        
    df = pd.read_csv('../data/' + df_name + '_preproc.csv', sep = ',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df


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
    graph.write_png("./results/trees/tree.png")

#Returns depth and leaves given a decision tree
def print_properties_tree(learner):
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    return depth, leaves

#Returns coefficients of the given logistic regression
def print_properties_lr(learner):
    return learner.coef_

#Classifier training
def train_model(df_name, seed):
    train = pd.read_csv('../data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    clf = DecisionTreeClassifier(random_state = seed)

    learner = clf.fit(X_train, y_train)
    return learner

# TODO: modificar esto
def save_model(learner, dataset_name, seed, variable_name, num_of_generations, num_of_individuals, individual_id, model, method, objectives):
    # save the model to disk
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "__" + objectives[i].__name__

    path = './results/' + str(method) + '/models/' + model + '/' + dataset_name + "/"
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








def generate_random_string(length):
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choices(letters_and_digits, k=length))

def test_and_save_results(x_test, y_test, prot_test, classifiers, objectives, generations, nind, ngen, dat, var, seed, obj, extra):
    
    save_name = ''
    obj_str = '_'.join(objectives) 
    extra_str = '_'.join(objectives)
    if not extra is None:
        save_name = 'results/' + dat + '__' + var + '__obj_' + obj_str + '__seed_' + str(seed) + '__extra_' + extra_str + '__nind_' + str(nind) + '__ngen_' + str(ngen) + '.csv'
    else:
        save_name = 'results/' + dat + '__' + var + '__obj_' + obj_str + '__seed_' + str(seed) + '__nind_' + str(nind) + '__ngen_' + str(ngen) + '.csv'
    
    dict_generate = {'ID': [], 'seed': [], 'creation_mode': []}
    
    for elem in obj:
        dict_generate[elem + '_val'] = []
    for elem in obj:
        dict_generate[elem + '_test'] = []
    
    dict_generate['repre'] = []
    
    for clf in classifiers:
        ID = generate_random_string(20)
        dict_generate['ID'].append(ID)
        dict_generate['seed'].append(seed)
        dict_generate['repre'].append(clf.repre_to_node_id())
        dict_generate['creation_mode'].append(clf.creation_mode)
        test_objs = clf.test_tree(x_test, y_test, prot_test)
        for i in range(len(obj)):
            dict_generate[obj[i] + '_val'].append(clf.objectives[i])
            dict_generate[obj[i] + '_test'].append(test_objs[i])
        
    df = pd.DataFrame(dict_generate)
    
    df.to_csv(save_name, index=False)
        



