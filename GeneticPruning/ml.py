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

def read_data(df_name):
    """
    Reads the dataset to work with. You have to work with preprocessed data, 
    and to ensure it, we will only read the files ending with _preproc
        Parameters:
            - df_name: Dataset name to read
        Returns:
            - df: DataFrame with the information read.
    """
    df = pd.read_csv(f"{PATH_TO_DATA}/{df_name}.csv", sep = ',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df





def get_matrices(df_name, y_col, seed):
    """
    Data preprocessing and splits dataframe into train and test
        Parameters:
            - y_col: Name of the attribute to predict
            - seed: Random seed to make the partition
        Returns:
            - X_train, X_val, X_test, y_train, y_val, y_test : Training, validation and tests set divided by X and y
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
    y = df.loc[:, y_col].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test





def write_train_val_test(df_name, prot_col, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Exports to csv files train, validation and test sets
        Parameters:
            - df_name: Dataset name
            - prot_col: Name of the protected column
            - seed: Random seed
            - X_train, X_val, X_test, y_train, y_val, y_test : Training, validation and tests set divided by X and y
    """
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_train_seed_{seed}.csv", index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_val_seed_{seed}.csv", index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv(f"{PATH_TO_DATA}/train_val_test_standard/{df_name}/{df_name}_{prot_col}_test_seed_{seed}.csv", index = False)





def print_tree(classifier, features):
    """
    Exports obtained decision tree to a png file
        Parameters:
            - classifier: Classifier to print
            - features: Features names
    """
    dot_data = StringIO()
    export_graphviz(classifier, out_file = dot_data, feature_names = features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("./results/trees/tree.png")





def print_properties_tree(learner):
    """
    Returns depth and leaves given a decision tree
        Parameters:
            - learner: model to get properties
        Returns:
            - depth: depth of the model
            - leaves: number of leaves of the model
            - w_avg_depth: depth weighted by the number of instances that fall into each leaf.
    """
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    w_avg_depth = data_weight_avg_depth(learner)
    return depth, leaves, w_avg_depth

def save_model(learner, seed, prot, num_of_generations, num_of_individuals, individual_id, objectives):
    """
    Saves the model to disk
        Parameters:
            - learner: Learner to save
            - seed: Random seed
            - prot: Name of the protected attribute
            - num_of_generations: Number of generations
            - num_of_individuals: Number of individuals
            - individual_id: ID of the individual
            - objectives: objectives function as a string
    """
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "__" + objectives[i].__name__

    path = f"{PATH_TO_RESULTS}/graphics/trees/"
    filename =  f"model_id_{individual_id}_seed_{seed}_var_{prot}_gen_{num_of_generations}_indiv_{num_of_individuals}_obj_{str_obj}.sav"
    pickle.dump(learner, open(path + filename, 'wb'))
    return





def val_model(X_val, learner):
    """
    Validates the classifier (comparison with validation set)
        Parameters:
            - X_val: Validation dataset
            - learner: learner to validate
        Returns:
            - results of validation
    """
    return learner.predict(X_val)






def test_model(X_test, learner):
    """
    Tests the classifier (comparison with test set)
        Parameters:
            - X_test: Test dataset
            - learner: learner to validate
        Returns:
            - results of test
    """
    return learner.predict(X_test)





def data_weight_avg_depth(learner):
    """
    Second complexity measure, that complements the first one.
    Return the weighted average of the depth of all leaves nodes,
    considering the number of training samples that fell on each one.
        Parameters:
            - learner: learner to calculate the measures
        Returns:
            - weighted average depth
    """
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
    """
    Generates a random string
        Parameters:
            - length: length of the string 
        Returns:
            - Random string
    """
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choices(letters_and_digits, k=length))





def create_gen_stats_df():
    """
    Creates the dataframe to store generations data
        Returns:
            - DataFrame for containing generations data 
    """
    store_dimensions = ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    
    new_dict = {}
    for elem in store_dimensions:
        new_dict[f"min_{elem}"] = []
        new_dict[f"mean_{elem}"] = []
        new_dict[f"max_{elem}"] = []
        new_dict[f"std_{elem}"] = []
    new_dict['process_time'] = []
    new_dict['total_time'] = []
    
    return pd.DataFrame(data=new_dict)





def update_gen_stats_df(store_df, newpop, p_time, t_time):
    """
    Updates the generations info dataframe to store info of a new generation
        Parameters:
            - store_df: Dataframe which stores generations data
            - newpop: Population of the new generation
            - p_time: Process time for creating the new population
            - t_time: Total time for creating the new population
        Returns:
            - Dataframe containing the updated information
    """
    store_dimensions = ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    
    dict_store = {}
    for dimension in store_dimensions:
        dict_store[dimension] = []
    
    for elem in newpop:
        elem.get_tree()
        dict_store['prunings'].append(elem.num_prunings)
        dict_store['leaves'].append(elem.num_leaves)
        dict_store['depth'].append(elem.depth)
        dict_store['data_avg_depth'].append(elem.data_avg_depth)
        dict_store['depth_unbalance'].append(elem.depth_unbalance)

    for dimension in store_dimensions:
        dict_store[dimension] = np.array(dict_store[dimension])

    new_row = {}
    for dimension in store_dimensions:
        new_row[f"min_{dimension}"] = dict_store[dimension].min()
        new_row[f"mean_{dimension}"] = dict_store[dimension].mean()
        new_row[f"max_{dimension}"] = dict_store[dimension].max()
        new_row[f"std_{dimension}"] = dict_store[dimension].std()
    new_row['process_time'] = p_time
    new_row['total_time'] = t_time
    
    return pd.concat([store_df, pd.DataFrame([new_row])], ignore_index=True)





def create_gen_population_df(obj):
    """
    Creates the Dataframe for storing individual information
        Parameters:
            - obj: Objectives to optimize
        Returns:
            - Dataframe for storing individuals information
    """
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': []}

    for elem in obj:
        dict_individual[elem + '_val'] = []

    size_features = ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    for elem in size_features:
        dict_individual[elem] = []
    
    return pd.DataFrame(dict_individual)





def update_gen_population(pop_df, new_pop, obj, seed):
    """
    Stores informations of new individuals into the population dataframe
        Parameters:
            - pop_df: Dataframe which stores information about individuals
            - new_pop: List containing new individuals for which to store information
            - obj: List of objective functions
            - seed: Random seed
        Returns:
            - Dataframe which stores information about individuals with updated information
    """
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': []}

    for elem in obj:
        dict_individual[elem + '_val'] = []

    size_features = ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    for elem in size_features:
        dict_individual[elem] = []
    
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
        
        dict_individual['prunings'].append(clf.num_prunings)
        dict_individual['leaves'].append(clf.num_leaves)
        dict_individual['depth'].append(clf.depth)
        dict_individual['data_avg_depth'].append(clf.data_avg_depth)
        dict_individual['depth_unbalance'].append(clf.depth_unbalance)

    return pd.concat([pop_df, pd.DataFrame(dict_individual)], ignore_index=True)






def test_and_save_results(x_test, y_test, prot_test, classifiers, gen_stats_df, population_df, nind, ngen, dat, var, seed, obj, extra):
    """
    Test considered models and store the results
        Parameters:
            - x_test: Predictor attributes for test datset
            - y_test: Attribute to predict for test set
            - prot_test: Protected attribute for test test
            - classifiers: Classifers to test and store
            - gen_stats_df: Dataframe containing information of the execution of each generation
            - population_df: Dataframe containing information of individuals
            - nind: Number of individuals contained in the population
            - ngen: Number of generations
            - dat: Dataset name
            - var: Name of the protected attribute
            - seed: Random seed
            - obj: List of objective functions
            - extra: Extra objective functions for which the process did optimized
    """
    save_population_name = save_gen_name = save_pareto_run_name = ''
    obj_str = '__'.join(obj)
    extra_str = ''
    if not extra is None:
        extra_str = '__'.join(extra)
        extra_str = '_ext_' + extra_str

    save_population_name = f"{PATH_TO_RESULTS}/population/{dat}/{dat}_seed_{seed}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv"
    save_gen_name = f"{PATH_TO_RESULTS}/generation_stats/{dat}/{dat}_seed_{seed}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv"
    save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}_seed_{seed}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv"
    
    dict_individual = {'ID': [], 'seed': [], 'creation_mode': []}
    
    for elem in obj:
        dict_individual[elem + '_val'] = []

    for elem in obj:
        dict_individual[elem + '_test'] = []
    
    size_features = ['prunings','leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
    for elem in size_features:
        dict_individual[elem] = []

    dict_individual['repre'] = []
    
    for clf in classifiers:
        ID = generate_random_string(20)
        dict_individual['ID'].append(ID)
        dict_individual['seed'].append(seed)
        dict_individual['repre'].append(clf.repre_to_node_id())
        dict_individual['creation_mode'].append(clf.creation_mode)

        test_objs = clf.test_tree(x_test, y_test, prot_test)
        for i in range(len(obj)):
            dict_individual[obj[i] + '_val'].append(clf.objectives[i])
            dict_individual[obj[i] + '_test'].append(test_objs[i])
        
        dict_individual['prunings'].append(clf.num_prunings)
        dict_individual['leaves'].append(clf.num_leaves)
        dict_individual['depth'].append(clf.depth)
        dict_individual['data_avg_depth'].append(clf.data_avg_depth)
        dict_individual['depth_unbalance'].append(clf.depth_unbalance)
        
    df = pd.DataFrame(dict_individual)
    
    df.to_csv(save_pareto_run_name, index=False)
    gen_stats_df.to_csv(save_gen_name, index=False)
    population_df.to_csv(save_population_name, index=False)



def correct_pareto_optimal(dat, var, objectives, nind, ngen, seed, i, extra, struc):

    obj_str = '__'.join(objectives) 
    extra_str = ''
    if not extra is None:
        extra_str = '__'.join(extra)
        extra_str = '_ext_' + extra_str

    all_indivs = []
    pareto_optimal = []
    pareto_fronts = pd.read_csv(f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}_seed_{seed + i}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv")
    hyperparameters = []
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

    pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}_seed_{seed + i}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv", index=False)

# TODO: CAMBIAR LA PARTE DE STRUC, CADA SEED TIENE UN STRUC
def calculate_pareto_optimal(dat, var, objectives, nind, ngen, seed, runs, extra, struc, correct=True):
    """
    Calculates Pareto optimal individuals from results of all runs
        Parameters:
            - dat: Dataset name
            - var: Protected attribute name
            - objectives: list of objective functions
            - nind: Number of individuals contained in the populatin
            - ngen: Number of generations
            - seed: Random seed
            - runs: Number of runs 
            - extra: Extra objective functions for which the process did optimized
            - struc: Tree structure of matrix tree
        Returns:
            - pareto_optimal: List of Pareto optimal individuals
            - pareto_optimal_df: Dataframe of Pareto optimal individuals
    """
    pareto_fronts = []
    all_indivs = []
    pareto_optimal = []
    obj_str = '__'.join(objectives) 
    extra_str = ''
    if not extra is None:
        extra_str = '__'.join(extra)
        extra_str = '_ext_' + extra_str
    
    for i in range(runs):
        if correct:
            correct_pareto_optimal(dat, var, objectives, nind, ngen, seed, i, extra, struc)
        save_pareto_run_name = f"{PATH_TO_RESULTS}/pareto_individuals/runs/{dat}/{dat}_seed_{seed + i}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv"
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

    pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}/pareto_individuals/overall/{dat}/{dat}_seed_{seed}_var_{var}_gen_{ngen}_indiv_{nind}_model_GP_obj_{obj_str}{extra_str}.csv", index=False)

    return pareto_optimal, pareto_optimal_df                   #Population of pareto front individuals