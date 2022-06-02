import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import random
import csv
import sys
import time
import warnings
import importlib
warnings.filterwarnings("ignore")

sys.path.append("..")
from general.ml import *
from general.problem import Problem
from general.qualitymeasures import *

nind = ngen = dat = var = bseed = nruns = obj = extra = False        #Possible parameters given

message = "\nExecutes a multiobjective evolutionary optimization algorithm to solve a problem, based on prunings over a base Decision Tree Classifier, with the following parameters:\n"

message += "\nThe following parameters have been given by the user:"

error = False
for i in range(1, len(sys.argv)):           #We're going to read all parameters
    valid = False

    message += ("\n- " + sys.argv[i])
    param = sys.argv[i].split('=')
    
    if not valid and param[0] == 'nind':              #Nunber of individuals for each population in the algorithm
        nind = valid = True
        individuals = int(param[1])
    
    if not valid and param[0] == 'ngen':              #Number of generations for the algorithm
        ngen = valid = True
        generations = int(param[1])
    
    if not valid and param[0] == 'dat':               #Name of the dataset
        dat = valid = True
        dataset = param[1]
    
    if not valid and param[0] == 'var':               #Name of the sensitive variable
        var = valid = True
        variable = param[1]
    
    if not valid and param[0] == 'bseed':             #Base seed for the 1st run of the algorithm
        bseed = valid = True
        set_seed_base = int(param[1])
    
    if not valid and param[0] == 'nruns':             #Number of runs to execute the algorithm
        nruns = valid = True
        n_runs = int(param[1])
    
    if not valid and param[0] == 'obj':               #Objectives to take into account that we'll try to miminize
        obj = valid = True
        objectives = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        objectives = [objdict[x] for x in objectives]
    
    if not valid and param[0] == 'extra':               #Objectives to calculate BUT NOT to be optimized
        extra = valid = True
        extraobj = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        extraobj = [objdict[x] for x in extraobj]

    if not valid and param[0] == 'help':              #The user wants help
        print('\nThis file contains the code for executing a multiobjective evolutionary optimisation algorithm, to a binary classification problem. Parameters accepted are:\n\n\
\t- nidn=(integer): Number of individuals to have each population during algorithm execution. The default is 50.\n\n\
\t- ngen=(integer): Number of generations for the algorithm to be executed (stopping criteria). The default is 300\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- var=(variable): Name of the sensitive variable for the dataset variable. Sensitive considered variables for each of the previous datasets are: adult-race, german-age, propublica_recidivism-race, propublica_violent_recidivism-race, ricci-Race. The default is the first variable of a the dataset (It is absolutely recommendable to change)\n\n\
\t- bseed=(integer): Base seed which will be used in the first run of the algorithm. It\'s used for train-validation-test split for the data, and other possible needs which require randomness. The default is 100.\n\n\
\t- nruns=(integer): Number of runs for the algorithm to be executed with different seeds. Each run takes consecutive seeds with respect to the previous one, starting from the base seed. The default is 10.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,dem_fpr.\n\n\
\t- extra=(comm separated list of objectives): List of objectives to not be used in optimization, but to be calculated in individuals generated. Possible values are the same as in obj. The default is None.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython main.py nind=60 ngen=300 dat=propublica_recidivism var=race bseed=100 nruns=10 obj=gmean_inv,dem_fpr,dem_ppv,num_leaves\n\n\
Results are saved into the corresponding results/(algorithm)/individuals folder. There are 2 kind of files:\n\n\
\t- individuals_... files: contain all individuals generated by the algorithm for a specific run.\n\n\
\t- individuals_pareto_... files: contain all Pareto optimal individuals generated by the algorithm for a specific run.\n\n')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print('At least 1 parameter name introduced is invalid.\n\
Please check out for mistakes there. Possible parameters are:\n\
nind, ngen, dat, var, bseed, nruns, extra.\n\
Type: \"python fairness.py help\" for help. Aborting\n')
        sys.exit(1)

print(message + "\n---")

    
#Now we're going to assign default values to each non intrudiced parameter

print("\nThe following parameters will be set by default:")
if not nind:
    individuals = 50        #Default number of individuals for each population in the algorithm: 50
    print("- nind=" + str(individuals))
if not ngen:              
    generations = 300       #Default number of generations for the algorithm: 300
    print("- ngen=" + str(generations))
if not dat:
    dataset = 'german'      #Default dataset: german
    print("- dat=" + dataset)
if not var:
    variable = pd.read_csv('../data/' + dataset + '.csv').columns[0]           #Default sensitive variable: First dataset variable
    print("- var=" + variable)
if not bseed:
    set_seed_base = 100             #Base seed for the 1st run of the algorithm
    print("- bseed=" + str(set_seed_base))
if not nruns:
    n_runs = 10                     #Number of runs to execute the algorithm
    print("- nruns=" + str(n_runs))
if not obj:
    objectives = [gmean_inv, dem_fpr] #Objectives to take into account that we'll try to miminize
    strobj = objectives[0].__name__
    for i in range(1, len(objectives)):
        strobj += "," + objectives[i].__name__
    print("- objectives=" + strobj)
if not extra:
    extraobj = None
    print("- extra= None")
print('---')


if len(objectives) < 2:
    print("There should be at least 2 objectives. Aborting.")
    sys.exit(1)

for run in range(n_runs):
    set_seed = set_seed_base + run

    # write datasets
    X_tr, X_v, X_tst, y_tr, y_v, y_tst = get_matrices(dataset, set_seed)
    write_train_val_test(dataset, set_seed, X_tr, X_v, X_tst, y_tr, y_v, y_tst)
    
    # number of rows in train
    num_rows_train = get_matrices(dataset, set_seed)[0].shape[0]

    #calculate_measures_save(pareto, algorithm, dataset, variable, objectives, set_seed, run, individuals, generations, ends[-1]-starts[-1], False)
    
    first = True
    for p in pareto:    
        problem.test_and_save(p,first,problem.seed, algorithm)
        first = False
    
#Calculate file with the general pareto front using all pareto fronts in every execution
print("Calculating pareto optimal solutions using all runs...")
pareto_optimal_exec, pareto_optimal_df = problem.calculate_pareto_optimal(set_seed_base, n_runs, algorithm)
print("Execution succesful!\n------------------------------")