from logging import makeLogRecord
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
import os

from ml import get_matrices, write_train_val_test, test_and_save_results, calculate_pareto_optimal
from genetic import Genetic_Pruning_Process_NSGA2
from individual import Tree_Structure

PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/results/GP'

nind = ngen = dat = sens_col = bseed = nruns = obj = extra = False        #Possible parameters given
dataset = None
datasetlist = ['academic','adult','arrhythmia','bank','catalunya','compas','credit','crime','default','diabetes-w','diabetes','drugs','dutch','german','heart','hrs','insurance','kdd-census','lsat','nursery','obesity', 'older-adults','oulad','parkinson','ricci','singles','student','tic','wine','synthetic-athlete','synthetic-disease','toy']
dict_outcomes = {'academic': 'atd','adult': 'income','arrhythmia': 'arrhythmia','bank': 'Subscribed','catalunya': 'recid','compas': 'score','credit': 'NoDefault','crime': 'ViolentCrimesPerPop','default': 'default','diabetes-w': 'Outcome','diabetes': 'readmitted','drugs': 'Coke','dutch': 'status','german': 'Label','heart': 'class','hrs': 'score','insurance': 'charges','kdd-census': 'Label','lsat':'ugpa','nursery': 'class','obesity': 'NObeyesdad','older-adults': 'mistakes','oulad': 'Grade','parkinson': 'total_UPDRS','ricci': 'Combine','singles': 'income','student': 'G3','tic': 'income', 'wine': 'quality','synthetic-athlete': 'Label','synthetic-disease': 'Label','toy': 'Label'}
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}

message = "\nExecutes a multiobjective evolutionary optimization algorithm to solve a problem, based on prunings over a base Decision Tree Classifier, with the following parameters:\n"
message += "\nThe following parameters have been given by the user:"
extraobj = None

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
        assert dataset in datasetlist
        sens_col = dict_protected[dataset]
        y_col = dict_outcomes[dataset]
        print("\n- y=" + y_col)
        print("\n- prot=" + sens_col)

    
    if not valid and param[0] == 'bseed':             #Base seed for the 1st run of the algorithm
        bseed = valid = True
        set_seed_base = int(param[1])
    
    if not valid and param[0] == 'nruns':             #Number of runs to execute the algorithm
        nruns = valid = True
        n_runs = int(param[1])
    
    if not valid and param[0] == 'obj':               #Objectives to take into account that we'll try to miminize
        obj = valid = True
        objectives = param[1].split(',')
        poss_obj = ['accuracy', 'fpr_diff', 'tpr_diff', 'ppv_diff', 'pnr_diff', 'gmean_inv']
        for x in objectives:
            if x not in poss_obj:
                print("You introduceded at least 1 objective which is not recognised.")
                print("Possible objectives are:" + ', '.join(poss_obj))
                sys.exit(-1)
    
    if not valid and param[0] == 'extra':               #Objectives to calculate BUT NOT to be optimized
        extra = valid = True
        extraobj = param[1].split(',')
        for x in extraobj:
            if x not in ['accuracy', 'fpr_diff']:
                print("You introduceded at least 1 objective which is not recognised.")
                print("Possible objectives are: accuracy, fpr_diff")
                sys.exit(-1)

    if not valid and param[0] == 'help':              #The user wants help
        print('\nThis file contains the code for executing a multiobjective evolutionary optimisation algorithm, to a binary classification problem. Parameters accepted are:\n\n\
\t- nidn=(integer): Number of individuals to have each population during algorithm execution. The default is 50.\n\n\
\t- ngen=(integer): Number of generations for the algorithm to be executed (stopping criteria). The default is 300\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- bseed=(integer): Base seed which will be used in the first run of the algorithm. It\'s used for train-validation-test split for the data, and other possible needs which require randomness. The default is 100.\n\n\
\t- nruns=(integer): Number of runs for the algorithm to be executed with different seeds. Each run takes consecutive seeds with respect to the previous one, starting from the base seed. The default is 10.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, fpr_diff, ppv_diff, pnr_diff, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,fpr_diff.\n\n\
\t- extra=(comm separated list of objectives): List of objectives to not be used in optimization, but to be calculated in individuals generated. Possible values are the same as in obj. The default is None.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython main.py nind=60 ngen=300 dat=propublica_recidivism var=race bseed=100 nruns=10 obj=gmean_inv,fpr_diff,ppv_diff,num_leaves\n\n\
Results are saved into the corresponding results/(algorithm)/individuals folder. There are 2 kind of files:\n\n\
\t- individuals_... files: contain all individuals generated by the algorithm for a specific run.\n\n\
\t- individuals_pareto_... files: contain all Pareto optimal individuals generated by the algorithm for a specific run.\n\n')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print(param[0])
        print('At least 1 parameter name introduced is invalid.\n\
Please check out for mistakes there. Possible parameters are:\n\
nind, ngen, dat, var, bseed, nruns, extra.\n\
Type: \"python fairness.py help\" for help. Aborting\n')
        sys.exit(1)

print(message + "\n---")

    
#Now we're going to assign default values to each non intrudiced parameter

print("\nThe following parameters will be set by default:")
if not nind:
    #individuals = 50        #Default number of individuals for each population in the algorithm: 50
    individuals = 50
    print("- nind=" + str(individuals))
if not ngen:              
    #generations = 300       #Default number of generations for the algorithm: 300
    generations = 300
    print("- ngen=" + str(generations))
if not dat:
    dataset = 'german'      #Default dataset: german
    sens_col = dict_protected[dataset]
    y_col = dict_outcomes[dataset]
    print("- dat=" + dataset)
    print("- y=" + dict_outcomes[dataset])
    print("- prot=" + dict_protected[dataset])
    
if not bseed:
    set_seed_base = 100             #Base seed for the 1st run of the algorithm
    print("- bseed=" + str(set_seed_base))
if not nruns:
    #n_runs = 10                     #Number of runs to execute the algorithm
    n_runs = 10
    print("- nruns=" + str(n_runs))
if not obj:
    objectives = ["gmean_inv", "fpr_diff"] #Objectives to take into account that we'll try to miminize
    strobj = objectives[0]
    for i in range(1, len(objectives)):
        strobj += "," + objectives[i]
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

    obj_str = '__'.join(objectives) 

    execute = False
    save_population_name = None
    extra_str = ''
    if not extraobj is None:
        extra_str = '__'.join(extraobj)

    save_population_name = f"{PATH_TO_RESULTS}/population/{dataset}/{dataset}_seed_{set_seed}_var_{sens_col}_gen_{generations}_indiv_{individuals}_model_GP_obj_{obj_str}{extra_str}.csv"
    #print(save_population_name)
    try:
        read = pd.read_csv(save_population_name)
        print(f"Result file for {dataset} dataset, seed {set_seed}, and the others parameters already existed!")
        if read.shape[0] < generations * individuals:
            execute = True
            print("\tBut it lacks some individuals, so it will be executed again")
    except FileNotFoundError:
        execute = True

    if execute:
        print("--- RUN:", run)
        # write datasets
        x_train, x_val, x_test, y_train, y_val, y_test = get_matrices(dataset, y_col, set_seed)
        write_train_val_test(dataset, sens_col, set_seed, x_train, x_val, x_test, y_train, y_val, y_test)
        x_train = x_train.loc[:, x_train.columns != 'y']
        x_val = x_val.loc[:, x_val.columns != 'y']
        x_test = x_test.loc[:, x_test.columns != 'y']

        prot_train = x_train[sens_col].astype(int)
        prot_val = x_val[sens_col].astype(int)
        prot_test = x_test[sens_col].astype(int)
        
        struc = Tree_Structure(x_train, y_train, prot_train, x_val, y_val, prot_val, run)

        gen_process = Genetic_Pruning_Process_NSGA2(struc, objectives, generations, individuals, 1, 0.2)

        indivs, gen_stats_df, population_df = gen_process.genetic_optimization(set_seed)
        
        """
        print(indivs)
        for indiv in indivs:    
            print(i)
            print(indiv.repre)
            for i in range(len(objectives)):
                print(objectives[i], indiv.objectives[i])
        """
        
        test_and_save_results(x_test, y_test, prot_test, indivs, gen_stats_df, population_df, individuals, generations, dataset, sens_col, set_seed, objectives, extraobj)


x_train, x_val, x_test, y_train, y_val, y_test = get_matrices(dataset, y_col, set_seed)
write_train_val_test(dataset, sens_col, set_seed, x_train, x_val, x_test, y_train, y_val, y_test)
x_train = x_train.loc[:, x_train.columns != 'y']
x_val = x_val.loc[:, x_val.columns != 'y']
x_test = x_test.loc[:, x_test.columns != 'y']

prot_train = x_train[sens_col].astype(int)
prot_val = x_val[sens_col].astype(int)
prot_test = x_test[sens_col].astype(int)

struc = Tree_Structure(x_train, y_train, prot_train, x_val, y_val, prot_val, run)

print("Calculating pareto optimal solutions using all runs...")
calculate_pareto_optimal(dataset, sens_col, objectives, individuals, generations, set_seed_base, n_runs, extraobj, struc)
print("Execution succesful!\n------------------------------")
    



"""
#Calculate file with the general pareto front using all pareto fronts in every execution
print("Calculating pareto optimal solutions using all runs...")
pareto_optimal_exec, pareto_optimal_df = problem.calculate_pareto_optimal(set_seed_base, n_runs, algorithm)
print("Execution succesful!\n------------------------------")
"""