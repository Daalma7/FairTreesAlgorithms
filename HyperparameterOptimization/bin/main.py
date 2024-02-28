import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import random
import csv
import sys
import time
import warnings
import importlib
import os
import re
warnings.filterwarnings("ignore")


PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))) + '/results/'

sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from general.ml import gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth, get_matrices, write_train_val_test
from general.problem import Problem

def clean_feature_name(name):
    # Replace special characters with underscores or any other safe character
    return name.replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('"', '_').replace(':', '_')


alg = nind = ngen = dat = sens_col = bseed = nruns = obj = mod = extra = False        #Possible parameters given
dataset = None
str_obj = str_extra = ''
datasetlist = ['academic','adult','arrhythmia','bank','catalunya','compas','credit','crime','default','diabetes-w','diabetes','drugs','dutch','german','heart','hrs','insurance','kdd-census','lsat','nursery','obesity', 'older-adults','oulad','parkinson','ricci','singles','student','tic','wine','synthetic-athlete','synthetic-disease','toy']
dict_outcomes = {'academic': 'atd','adult': 'income','arrhythmia': 'arrhythmia','bank': 'Subscribed','catalunya': 'recid','compas': 'score','credit': 'NoDefault','crime': 'ViolentCrimesPerPop','default': 'default','diabetes-w': 'Outcome','diabetes': 'readmitted','drugs': 'Coke','dutch': 'status','german': 'Label','heart': 'class','hrs': 'score','insurance': 'charges','kdd-census': 'Label','lsat':'ugpa','nursery': 'class','obesity': 'NObeyesdad','older-adults': 'mistakes','oulad': 'Grade','parkinson': 'total_UPDRS','ricci': 'Combine','singles': 'income','student': 'G3','tic': 'income', 'wine': 'quality','synthetic-athlete': 'Label','synthetic-disease': 'Label','toy': 'Label'}
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}

message = "\nExecutes a multiobjective evolutionary optimization algorithm to solve a problem, with the following parameters:\n"
message += "\nThe following parameters have been given by the user:"

error = False
for i in range(1, len(sys.argv)):           #We're going to read all parameters
    valid = False

    message += ("\n- " + sys.argv[i])
    param = sys.argv[i].split('=')
    
    if not valid and param[0] == 'alg':               #Algorithm to use
        alg = valid = True
        algorithm = param[1]
        Evomodule = importlib.import_module("algorithms." + algorithm + ".evolution")
    
    if not valid and param[0] == 'nind':              #Nunber of individuals for each population in the algorithm
        nind = valid = True
        individuals = int(param[1])
    
    if not valid and param[0] == 'ngen':              #Number of generations for the algorithm
        ngen = valid = True
        generations = int(param[1])
    
    if not valid and param[0] == 'dat':               #Name of the dataset
        dat = valid = True
        dataset = param[1]
        assert dataset in datasetlist
        sens_col = dict_protected[dataset]
        y_col = dict_outcomes[dataset]
        message += "\n- y=" + y_col
        message += "\n- prot=" + sens_col
    
    if not valid and param[0] == 'bseed':             #Base seed for the 1st run of the algorithm
        bseed = valid = True
        set_seed_base = int(param[1])
    
    if not valid and param[0] == 'nruns':             #Number of runs to execute the algorithm
        nruns = valid = True
        n_runs = int(param[1])
    
    if not valid and param[0] == 'model':             #Number of runs to execute the algorithm
        mod = valid = True
        model = param[1]
    
    if not valid and param[0] == 'obj':               #Objectives to take into account that we'll try to miminize
        obj = valid = True
        objectives = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        objectives = [objdict[x] for x in objectives]
        str_obj = objectives[0].__name__
        for i in range(1, len(objectives)):
            str_obj += "__" + objectives[i].__name__
    
    if not valid and param[0] == 'extra':               #Objectives to calculate BUT NOT to be optimized
        extra = valid = True
        extraobj = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        extraobj = [objdict[x] for x in extraobj]
        str_extra = '_ext_' + extraobj[0].__name__
        for i in range(1, len(extraobj)):
            str_extra += "__" + extraobj[i].__name__

    if not valid and param[0] == 'help':              #The user wants help
        print('\nThis file contains the code for executing a multiobjective evolutionary optimisation algorithm, to a binary classification problem. Parameters accepted are:\n\n\
\t- alg=(algorithm): Algorithm to be executed. Possible algorithms are nsga2, smsemoa and grea. The default is nsga2\n\n\
\t- nidn=(integer): Number of individuals to have each population during algorithm execution. The default is 50.\n\n\
\t- ngen=(integer): Number of generations for the algorithm to be executed (stopping criteria). The default is 300\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- bseed=(integer): Base seed which will be used in the first run of the algorithm. It\'s used for train-validation-test split for the data, and other possible needs which require randomness. The default is 100.\n\n\
\t- nruns=(integer): Number of runs for the algorithm to be executed with different seeds. Each run takes consecutive seeds with respect to the previous one, starting from the base seed. The default is 10.\n\n\
\t- model=(model abbreviation): Model to use. Possible models are Decision Tree (DT), Fair Decision Trees and Logistic Regression (LR). The default is DT.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,dem_fpr.\n\n\
\t- extra=(comm separated list of objectives): List of objectives to not be used in optimization, but to be calculated in individuals generated. Possible values are the same as in obj. The default is None.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython fairness.py nind=60 ngen=300 alg=nsga2 dat=propublica_recidivism bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves\n\n\
Results are saved into the corresponding results/(algorithm)/individuals folder. There are 2 kind of files:\n\n\
\t- individuals_... files: contain all individuals generated by the algorithm for a specific run.\n\n\
\t- individuals_pareto_... files: contain all Pareto optimal individuals generated by the algorithm for a specific run.\n\n')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print('At least 1 parameter name introduced is invalid.\n\
Please check out for mistakes there. Possible parameters are:\n\
alg, nind, ngen, dat, bseed, nruns, extra.\n\
Type: \"python fairness.py help\" for help. Aborting\n')
        sys.exit(1)

print(message + "\n---")

    
#Now we're going to assign default values to each non intrudiced parameter

print("\nThe following parameters will be set by default:")
if not alg:
    algorithm = 'nsga2'
    from algorithms.nsga2.evolution import Evolution    #Default algorithm: nsga2
    print("- alg=" + algorithm)
if not nind:
    individuals = 50        #Default number of individuals for each population in the algorithm: 50
    print("- nind=" + str(individuals))
if not ngen:              
    generations = 300       #Default number of generations for the algorithm: 300
    print("- ngen=" + str(generations))
if not dat:
    dataset = 'german'      #Default dataset: german
    sens_col = dict_protected[dataset]
    y_col = dict_outcomes[dataset]
    print("- dat=" + dataset)
    print("- y=" + y_col)
    print("- prot=" + sens_col)
if not bseed:
    set_seed_base = 100             #Base seed for the 1st run of the algorithm
    print("- bseed=" + str(set_seed_base))
if not nruns:
    n_runs = 10                     #Number of runs to execute the algorithm
    print("- nruns=" + str(n_runs))
if not mod:
    model = "DT"
    print("- model=" + model)
if not obj:
    objectives = [gmean_inv, dem_fpr] #Objectives to take into account that we'll try to miminize
    str_obj = objectives[0].__name__
    for i in range(1, len(objectives)):
        str_obj += "_" + objectives[i].__name__
    print("- objectives=" + str_obj)
if not extra:
    extraobj = None
    print("- extra= None")
print('---')

#We're going to get rid of incompatible model and objectives
if model == "LR" and (num_leaves in objectives or data_weight_avg_depth in objectives):
    print("There's an incompatibility between the model used and one of the objectives.\
\nThat objective can't be computed as depends on the model used. Please check it out.\n")
    sys.exit(1)

if len(objectives) < 2:
    print("There should be at least 2 objectives. Aborting.")
    sys.exit(1)


# Range of hyperparameters
variables_range = None

if model == "DT":               #If we're using Decision Trees:
    min_range_criterion = 0         #Gini
    max_range_criterion = 1         #Entropy
    
    min_range_max_depth = 3
    max_range_max_depth = None
    
    min_range_samples_split = 2
    max_range_samples_split = 40
    
    min_range_samples_leaf = 1
    max_range_samples_leaf = 60
    
    min_range_leaf_nodes = 2
    max_range_leaf_nodes = None
    
    min_range_class_weight = 1
    max_range_class_weight = 9

    variables_range = [(min_range_criterion, max_range_criterion),(min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_class_weight, max_range_class_weight)]


if model == "FDT":               #If we're using Fair Decision Trees:
    min_range_criterion = 0         #Gini
    max_range_criterion = 1         #Entropy
    
    min_range_max_depth = 3
    max_range_max_depth = None
    
    min_range_samples_split = 2
    max_range_samples_split = 40
    
    min_range_samples_leaf = 1
    max_range_samples_leaf = 60
    
    min_range_leaf_nodes = 2
    max_range_leaf_nodes = None
    
    min_range_class_weight = 1
    max_range_class_weight = 9

    min_fair_param = 0
    max_fair_param = 10

    variables_range = [(min_range_criterion, max_range_criterion),(min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_class_weight, max_range_class_weight), (min_fair_param, max_fair_param)]


if model == "LR":               #In case we've devided to use Logistic Regression
    min_range_max_iter = 20     #Maximum number of iterations taken for the solvers to converge (one stopping criteria)
    max_range_max_iter = 200

    min_range_tol = 0.0001
    max_range_tol = 0.1

    min_range_C = 0.001
    max_range_C = 100000

    min_range_l1_ratio = 0
    max_range_l1_ratio = 1

    min_range_class_weight = 1
    max_range_class_weight = 9

    variables_range = [(min_range_max_iter, max_range_max_iter),(min_range_tol, max_range_tol),(min_range_C, max_range_C),(min_range_l1_ratio, max_range_l1_ratio),(min_range_class_weight,max_range_class_weight)]

if model == "FLGBM":               #In case we've devided to use Logistic Regression
    min_range_lamb = 0
    max_range_lamb = 1
    
    min_range_num_leaves = 2
    max_range_num_leaves = 62
    
    min_range_min_data_in_leaf = 2
    max_range_min_data_in_leaf = 40
    
    min_range_max_depth = 2
    max_range_max_depth = None
    
    min_range_learning_rate = 0.01
    max_range_learning_rate = 0.2
    
    min_range_n_estimators = 50
    max_range_n_estimators = 200
    
    min_range_feature_fraction = 0.1
    max_range_feature_fraction = 1.0

    variables_range = [(min_range_lamb, max_range_lamb), (min_range_num_leaves, max_range_num_leaves), (min_range_min_data_in_leaf, max_range_min_data_in_leaf),(min_range_max_depth, max_range_max_depth),(min_range_learning_rate, max_range_learning_rate),(min_range_n_estimators, max_range_n_estimators),(min_range_feature_fraction, max_range_feature_fraction)]
    


for run in range(n_runs):

    set_seed = set_seed_base + run
    execute = False
    print(f"{PATH_TO_RESULTS}{model}/{algorithm}/population/{dataset}/{dataset}_seed_{set_seed}_var_{sens_col}_gen_{generations}_indiv_{individuals}_model_{model}_obj_{str_obj}{str_extra}.csv")
    try:
        read = pd.read_csv(f"{PATH_TO_RESULTS}{model}/{algorithm}/population/{dataset}/{dataset}_seed_{set_seed}_var_{sens_col}_gen_{generations}_indiv_{individuals}_model_{model}_obj_{str_obj}{str_extra}.csv")
        print(f"Result file for {dataset} dataset, seed {set_seed}, and the others parameters already existed!")
        if read.shape[0] < generations * individuals:
            execute = True
            print("\tBut it lacks some individuals, so it will be executed again")
    except FileNotFoundError:
        execute = True

    if execute:

        # write datasets
        X_tr, X_v, X_tst, y_tr, y_v, y_tst = get_matrices(dataset, y_col, set_seed)
        if model == 'FLGBM':
            for dat in [X_tr, X_v, X_tst]:
                dat.columns = [clean_feature_name(name) for name in dat.columns]


        write_train_val_test(dataset, sens_col, set_seed, X_tr, X_v, X_tst, y_tr, y_v, y_tst)
        
        # number of rows in train
        num_rows_train = get_matrices(dataset, y_col, set_seed)[0].shape[0]
        
        problem = Problem(num_of_variables = 5,
                        objectives = objectives,
                        extra=extraobj,
                        variables_range = variables_range,
                        individuals_df = pd.DataFrame(),
                        num_of_generations = generations,
                        num_of_individuals = individuals,
                        dataset_name = dataset,
                        variable_name = sens_col,
                        model = model,
                        seed = set_seed)

        print("------------RUN:",run)
        
        evo = Evomodule.Evolution(problem,
                        evolutions_df = pd.DataFrame(),
                        dataset_name = dataset,
                        model_name = model,
                        protected_variable = sens_col,
                        num_of_generations = generations,
                        num_of_individuals = individuals)
        
        pareto = evo.evolve()

        #calculate_measures_save(pareto, algorithm, dataset, variable, objectives, set_seed, run, individuals, generations, ends[-1]-starts[-1], False)
        
        first = True
        for p in pareto:    
            problem.test_and_save(p,first,problem.seed, algorithm)
            first = False


problem = Problem(num_of_variables = 5,
                objectives = objectives,
                extra=extraobj,
                variables_range = variables_range,
                individuals_df = pd.DataFrame(),
                num_of_generations = generations,
                num_of_individuals = individuals,
                dataset_name = dataset,
                variable_name = sens_col,
                model = model,
                seed = set_seed)
#Calculate file with the general pareto front using all pareto fronts in every execution
print("Calculating pareto optimal solutions using all runs...")
pareto_optimal_exec, pareto_optimal_df = problem.calculate_pareto_optimal(set_seed_base, n_runs, algorithm)
print("Execution succesful!\n------------------------------")