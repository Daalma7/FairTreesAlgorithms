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
import os
import numpy as np
from collections import OrderedDict as od
from individual import Individual_NSGA2


PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/results/GP'
PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/datasets/data'

#Reads the dataset to work with. You have to work with preprocessed data, and to ensure it, we will only read the files ending with _preproc
def read_data(df_name):
    df = pd.read_csv(f"{PATH_TO_DATA}/{df_name}.csv", sep = ',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df


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
    y = df.loc[:, y_col].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

#Exports to csv files train, validation and test sets
def write_train_val_test(df_name, prot_col, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_train_seed_{seed}.csv", index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_val_seed_{seed}.csv", index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_test_seed_{seed}.csv", index = False)

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
    w_avg_depth = data_weight_avg_depth(learner)
    return depth, leaves, w_avg_depth

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

#TODO Revisar que se ha hecho bien
#Validates the classifier (comparison with validation set)
def val_model(X_val, learner):
    return learner.predict(X_val)

#TODO Revisar que se ha hecho bien
#Tests the classifier (comparison with test set)
def test_model(X_test, learner):
    return learner.predict(X_test)

#Second complexity measure, that complements the first one.
#Return the weighted average of the depth of all leaves nodes, considering the number of training samples that fell on each one.
def data_weight_avg_depth(learner):
    stack = [(0,0)]                     #Root node id and its depth
    total_w_depth = 0.0
    tree = learner.tree_
    while len(stack) > 0:
        current_node, current_depth = stack.pop()
        if(learner.tree_.children_left[current_node] != learner.tree_.children_right[current_node]):    #If it's not a leaf
            stack.append((learner.tree_.children_left[current_node], current_depth + 1))    #Append both children with their depth increased
            stack.append((learner.tree_.children_right[current_node], current_depth + 1))
        else:
            weighted_samples = tree.weighted_n_node_samples[current_node]
            total_w_depth += weighted_samples * current_depth
    return total_w_depth


def generate_random_string(length):
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choices(letters_and_digits, k=length))


def create_gen_stats_df():
    return pd.DataFrame({'df_min_leaves':[], 'df_avg_leaves':[], 'df_max_leaves':[], 'df_std_leaves':[],
                  'df_min_depth':[], 'df_avg_depth':[], 'df_max_depth':[], 'df_std_depth':[],
                  'df_min_data_avg_depth':[], 'df_mean_data_avg_depth':[], 'df_max_data_avg_depth':[], 'df_std_data_avg_depth':[],
                  'process_time':[], 'total_time':[]})

def update_gen_stats_df(store_df, newpop, p_time, t_time):
    df_num_leaves = []
    df_depth = []
    df_data_avg_depth = []
    for elem in newpop:
        elem.get_tree()
        df_num_leaves.append(elem.num_leaves)
        df_depth.append(elem.depth)
        df_data_avg_depth.append(elem.data_avg_depth)

    df_num_leaves = np.array(df_num_leaves)
    df_depth = np.array(df_depth)
    df_data_avg_depth = np.array(df_data_avg_depth)

    c_df = pd.DataFrame({'df_min_leaves':[df_num_leaves.min()], 'df_avg_leaves':[df_num_leaves.mean()], 'df_max_leaves':[df_num_leaves.max()], 'df_std_leaves':[df_num_leaves.std()],
                  'df_min_depth':[df_depth.min()], 'df_avg_depth':[df_depth.mean()], 'df_max_depth':[df_depth.max()], 'df_std_depth':[df_depth.std()],
                  'df_min_data_avg_depth':[df_data_avg_depth.min()], 'df_mean_data_avg_depth':[df_data_avg_depth.mean()], 'df_max_data_avg_depth':[df_data_avg_depth.max()], 'df_std_data_avg_depth':[df_data_avg_depth.std()],
                  'process_time': [p_time], 'total_time':[t_time]})
    
    return pd.concat([store_df,c_df], ignore_index=True)


def create_gen_population_df(obj, seed):
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': [], 'num_prunings': [], 'num_leaves': [], 'depth':[], 'data_avg_depth':[], 'unbalance':[]}
    for elem in obj:
        dict_individual[elem + '_val'] = []
    return pd.DataFrame(dict_individual)

def update_gen_population(pop_df, new_pop, obj, seed):
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': [], 'num_prunings': [], 'num_leaves': [], 'depth':[], 'data_avg_depth':[], 'unbalance':[]}
    for elem in obj:
        dict_individual[elem + '_val'] = []
    dict_individual['repre'] = []

    for clf in new_pop:
        # TODO: Modificar el ID para que sea una propiedad del individuo en cuestión
        ID = generate_random_string(20)
        dict_individual['ID'].append(ID)
        dict_individual['seed'].append(seed)
        dict_individual['repre'].append(clf.repre_to_node_id())
        dict_individual['creation_mode'].append(clf.creation_mode)

        for i in range(len(obj)):
            dict_individual[obj[i] + '_val'].append(clf.objectives[i])
        
        dict_individual['num_prunings'].append(clf.num_prunings)
        dict_individual['num_leaves'].append(clf.num_leaves)
        dict_individual['depth'].append(clf.depth)
        dict_individual['data_avg_depth'].append(clf.data_avg_depth)
        dict_individual['unbalance'].append(clf.unbalance)

    return pd.concat([pop_df, pd.DataFrame(dict_individual)], ignore_index=True)






def test_and_save_results(x_test, y_test, prot_test, classifiers, gen_stats_df, population_df, objectives, nind, ngen, dat, var, seed, obj, extra):
    
    save_population_name = save_gen_name = save_pareto_run_name = ''
    obj_str = '_'.join(objectives) 
    if extra is None:
        save_population_name = f"{PATH_TO_RESULTS}/population/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__nind_{nind}__ngen_{ngen}.csv"
        save_gen_name = f"{PATH_TO_RESULTS}/generation_stats/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__nind_{nind}__ngen_{ngen}.csv"
        save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__nind_{nind}__ngen_{ngen}.csv"
    else:
        extra_str = '_'.join(extra)
        save_population_name = f"{PATH_TO_RESULTS}/population/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__extra_{extra_str}__nind_{nind}__ngen_{ngen}.csv"
        save_gen_name = f"{PATH_TO_RESULTS}/generation_stats/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__extra_{extra_str}__nind_{nind}__ngen_{ngen}.csv"
        save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}__{var}__obj_{obj_str}__seed_{seed}__extra_{extra_str}__nind_{nind}__ngen_{ngen}.csv"
        
    
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': [], 'num_prunings': [], 'num_leaves': [], 'depth':[], 'data_avg_depth':[], 'unbalance':[]}
    
    for elem in obj:
        dict_individual[elem + '_val'] = []
    for elem in obj:
        dict_individual[elem + '_test'] = []
    
    dict_individual['repre'] = []
    
    for clf in classifiers:
        print("----")
        print(clf)
        ID = generate_random_string(20)
        dict_individual['ID'].append(ID)
        dict_individual['seed'].append(seed)
        dict_individual['repre'].append(clf.repre_to_node_id())
        dict_individual['creation_mode'].append(clf.creation_mode)

        test_objs = clf.test_tree(x_test, y_test, prot_test)
        for i in range(len(obj)):
            dict_individual[obj[i] + '_val'].append(clf.objectives[i])
            dict_individual[obj[i] + '_test'].append(test_objs[i])
        
        dict_individual['num_prunings'].append(clf.num_prunings)
        dict_individual['num_leaves'].append(clf.num_leaves)
        dict_individual['depth'].append(clf.depth)
        dict_individual['data_avg_depth'].append(clf.data_avg_depth)
        dict_individual['unbalance'].append(clf.unbalance)
        
    df = pd.DataFrame(dict_individual)
    
    df.to_csv(save_pareto_run_name, index=False)
    gen_stats_df.to_csv(save_gen_name, index=False)
    population_df.to_csv(save_population_name, index=False)



def calculate_pareto_optimal(dataset, var, objectives, nind, ngen, seed_base, runs, extra, struc):
    pareto_fronts = []
    all_indivs = []
    pareto_optimal = []
    obj_str = '_'.join(objectives) 

    for i in range(runs):
        save_pareto_run_name = ''
        if extra is None:
            save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dataset}/{dataset}__{var}__obj_{obj_str}__seed_{seed_base + i}__nind_{nind}__ngen_{ngen}.csv"
        else:
            extra_str = '_'.join(extra)
            save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dataset}/{dataset}__{var}__obj_{obj_str}__seed_{seed_base + i}__extra_{extra_str}__nind_{nind}__ngen_{ngen}.csv"
        read = pd.read_csv(save_pareto_run_name)
        pareto_fronts.append(read)

    hyperparameters = []
    pareto_fronts = pd.concat(pareto_fronts)                            #Union of all pareto fronts got in each run
    pareto_fronts.reset_index(drop=True, inplace=True)                  #Reset index because for each run all rows have repeated ones
    for index, row in pareto_fronts.iterrows():                         #We create an individual object associated with each row
        indiv = Individual_NSGA2(struc, objectives, row['repre'], row['creation_mode'], [float(row[f"{obj}_val"]) for obj in objectives])
        hyperparameters = ['repre']
        indiv.features = [row[x] for x in hyperparameters]
        indiv.id = row['ID']
        indiv.domination_count = 0
        indiv.features = od(zip(hyperparameters, indiv.features))
        indiv.objectives = []
        for obj in objectives:
            if not obj == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                indiv.objectives.append(float(row[f"{obj}_val"]))
            else:                                   #In other case
                indiv.objectives.append(float(row[f"{obj}_val"]) / pareto_fronts[f"{obj}_val"].max())
        #The same with extra objectives
        indiv.extra = []
        if not extra is None: 
            for ext in extra:
                # We will insert all objectives, normalizing every objective that should be
                if not extra == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                    indiv.extra.append(float(row[f"{ext}_val"]))
                else:                                   #In other case
                    indiv.extra.append(float(row[f"{ext}_val"]) / pareto_fronts[f"{ext}_val"].max())
        indiv.creation_mode = row['creation_mode']
        all_indivs.append(indiv)
    for indiv in all_indivs:                       #Now we calculate all the individuals non dominated by any other (pareto front)
        indiv.domination_count == 0
        for other_indiv in all_indivs:
            if other_indiv.dominates(indiv):                
                indiv.domination_count += 1                        #Indiv is dominated by the second
        if indiv.domination_count == 0:                            #Could be done easily more efficiently, but could be interesting 
            pareto_optimal.append(indiv)
    pareto_optimal_df = []
    for p in pareto_optimal:                #We select individuals from the files corresponding to the pareto front ones (we filter by id)
        curr_id = p.id                      #BUT IF THERE ARE MORE THAN 1 INDIVIDUAL WITH THE SAME ID THEY WILL ALL BE ADDED, EVEN THOUGHT ONLY 1 OF THEM IS A PARETO OPTIMAL SOLUTION
        found = False                       #Which is by the way really unlikely since there are 36^10 possibilities for an id
        for index, row in pareto_fronts.iterrows():
            if row['ID'] == curr_id:
                pareto_optimal_df.append(pd.DataFrame({x : row[x] for x in pareto_fronts.columns.tolist()}, index=[0])) #We introduce here the not-normalized version of them
                found = True
        if not found:
            pareto_optimal.remove(p)
    #We extract them to a file
    pareto_optimal_df = pd.concat(pareto_optimal_df)
    pareto_optimal_df = pareto_optimal_df.drop_duplicates(subset=(['seed']+hyperparameters), keep='first')
    if extra is None:
        pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}/pareto_individuals/overall/{dataset}/{dataset}__{var}__obj_{obj_str}__seed_{seed_base + i}__nind_{nind}__ngen_{ngen}.csv", index = False, header = True, columns = list(pareto_fronts.keys()))
    else:
        pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}/pareto_individuals/overall/{dataset}/{dataset}__{var}__obj_{obj_str}__seed_{seed_base + i}__extra_{extra_str}__nind_{nind}__ngen_{ngen}.csv", index = False, header = True, columns = list(pareto_fronts.keys()))

    return pareto_optimal, pareto_optimal_df                   #Population of pareto front individuals
