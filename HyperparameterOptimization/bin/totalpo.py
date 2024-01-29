import pandas as pd
from math import ceil
import random
import csv
import sys
import warnings
import importlib
import os
import re
import numpy as np
from collections import OrderedDict as od
warnings.filterwarnings("ignore")

sys.path.append("..")
from general.ml import *
from general.individual import *

alg = dat = var = obj = mod = extra = False        #Possible parameters given

message = "\nCalculating pareto optimal individuals using the results of all executions using the following parameters:\n"

message += "\nThe following parameters have been given by the user:"
error = False
for i in range(1, len(sys.argv)):           #We're going to read all parameters
    valid = False

    message += ("\n- " + sys.argv[i])
    param = sys.argv[i].split('=')
    
    if not valid and param[0] == 'alg':               #Algorithms to use
        alg = valid = True
        algorithm = param[1].split(',')

    if not valid and param[0] == 'dat':               #Database name
        dat = valid = True
        dataset = param[1]
    
    if not valid and param[0] == 'var':               #Sensitive attribute name
        variable = param[1]
    
    if not valid and param[0] == 'model':             #Number of runs to execute the algorithm
        mod = valid = True
        model = param[1]
    
    if not valid and param[0] == 'obj':               #Objectives to take into account that we'll try to miminize
        obj = valid = True
        objectives = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        objectives = [objdict[x] for x in objectives]
    
    if not valid and param[0] == 'extra':               #Objectives to calculate BUT NOT to be optimized.
        extra = valid = True
        extraobj = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        extraobj = [objdict[x] for x in extraobj]
    
    if not valid and param[0] == 'help':              #The user wants help
        print('\nThis file contains the code to get all previous results from the executions of fairness.py with specific parameters and calculate the general pareto front for that problem and objectives. Parameters accepted are:\n\n\
\t- alg=(comm separated list of algorithms): Algorithms from which to take results. Possible algorithms are nsga2, smsemoa and grea. The default is nsga2,smsemoa,grea\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- var=(variable): Name of the sensitive variable for the dataset variable. Sensitive considered variables for each of the previous datasets are: adult-race, german-age, propublica_recidivism-race, propublica_violent_recidivism-race, ricci-Race. The default is the first variable of a the dataset (It is absolutely recommendable to change)\n\n\
\t- model=(model abbreviation): Model to use. Possible models are Decision Tree (DT), Fair Decision Tree (FDT) and Logistic Regression (LR). The default is DT.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,dem_fpr. IMPORTANT: Objectives should be written in the same order as they were written at the fairness.py execution sentence.\n\n\
\t- extra=(comm separated list of objectives): List of objectives to not be used in optimization, but to be calculated in individuals generated. Possible values are the same as in obj. The default is None. IMPORTANT: Objectives should be written in the same order as they were written at the fairness.py execution sentence.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython totalpo.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves\n\n\
Results are saved into results/general_pareto_fronts folder.\n\n')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print('There\'s at least 1 invalid introduced parameter.\n\
Please check out for mistakes there. Possible parameters are:\n\
alg, nind, ngen, dat, var, bseed, nruns\n')
        sys.exit(1)

print(message + "\n---")
    
#Now we're going to assign default values to each non intrudiced parameter

print("\nThe following parameters will be set by default:")
if not alg:
    algorithm = ['nsga2', 'smsemoa', 'grea']
    stralg = algorithm[0]
    for i in range(1, len(algorithm)):
        stralg += "," + algorithm[i]
    print("- alg=" + stralg)
if not dat:
    dataset = 'german'      #Default dataset: german
    print("- dat=" + dataset)
if not var:
    variable = pd.read_csv('../data/' + dataset + '.csv').columns[0]           #Default sensitive variable: First dataset variable
    print("- var=" + variable)
if not mod:
    model = "DT"
    print("- model=" + model)
if not obj:
    objectives = [gmean_inv, dem_fpr] #Objectives to take into account that we'll try to miminize
    strobj = objectives[0].__name__
    for i in range(1, len(objectives)):
        strobj += "," + objectives[i].__name__
    print("- objectives=" + strobj)
if not extra:
    extraobj = None
    print("- extra=None")
print('---')

#We're going to get rid of incompatible model and objectives
if model == "LR" and (num_leaves in objectives or data_weight_avg_depth in objectives):
    print("There's an incompatibility between the model used and one of the objectives.\
\nThat objective can't be computed as depends on the model used. Please check it out.\n")
    sys.exit(1)


#We're going to determine all pareto fronts found in all executions by all algorithms

str_obj = objectives[0].__name__
for i in range(1, len(objectives)):
    str_obj = str_obj + '__' + str(objectives[i].__name__)

if not extra:
    str_extra = ''
else:
    str_extra = '_ext_' + extraobj[0].__name__
    for i in range(1, len(extraobj)):
        str_extra = str_extra + '__' + str(extraobj[i].__name__)

if not extra:
    regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".csv")
else:
    regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".*" + str_extra + ".csv")

pareto_files = []           #Files where pareto optimal individuals have been stored for each execution of the algorithm with same parameters as provided
pareto_alg = []             #Algorithm used to generate each one of those files

for x in algorithm:
    rootdir = "../results/" + x + "/individuals"
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                pareto_files.append(rootdir + "/" + file)
                pareto_alg.append(x)


#Add the new pareto front solutions to the total pareto front solutions in order to improve it
print("\nAdd solutions found to general pareto optimal solutions for the problem...")

all_indivs = []
new_pareto_indivs = []
pareto_optimal_df = []

objectives_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst', 'num_leaves': 'num_leaves_tst', 'data_weight_avg_depth': 'data_weight_avg_depth_tst'}

#Now we will try to read the overall pareto front file for that dataset and objectives. If it doesn't exists, creates one with the pareto front argument.
pareto_df = []
for i in range(len(pareto_files)):
    print("-> Reading data from file: " + pareto_files[i])
    pareto_df.append(pd.read_csv(pareto_files[i]))                                                                      #We're going to read that file
    pareto_df[-1].insert(loc=1, column='algorithm', value=np.full(pareto_df[-1].shape[0], pareto_alg[i]).tolist())      #In addition, we're going to add a column representing the algorithm used

pareto_df = pd.concat(pareto_df) #Union of all dataframes

#Before continuing we need to get rid of duplicates using the id column, as the next code cant deal with them
pareto_df = pareto_df.drop_duplicates(subset=['id'], keep = 'first')

print("\nExtracting individuals...")

# Getting all individuals 
for index, row in pareto_df.iterrows():                         #We create an individual object associated with each row
    if model == "DT":
        indiv = IndividualDT()
        indiv.features = [float(row['criterion']), row['max_depth'], row['min_samples_split'], row['max_leaf_nodes'], row['class_weight']]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        indiv.actual_depth = row['actual_depth']
        indiv.actual_leaves = row['actual_leaves']
    if model == "FDT":
        indiv = IndividualDT()
        indiv.features = [float(row['criterion']), row['max_depth'], row['min_samples_split'], row['max_leaf_nodes'], row['class_weight'], row['fair_param']]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        indiv.actual_depth = row['actual_depth']
        indiv.actual_leaves = row['actual_leaves']
    if model == "LR":
        indiv = IndividualLR()
        indiv.features = [row['max_iter'], row['tol'], row['lambda'], row['l1_ratio'], row['class_weight']]
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
    indiv.id = row['id']
    indiv.domination_count = 0
    indiv.features = od(zip(hyperparameters, indiv.features))
    indiv.objectives = []
    indiv.extra = []
    for x in objectives:
        obj = objectives_dict.get(x.__name__)
        indiv.objectives.append(float(row[obj]))
    if extra:
        for x in extraobj:
            ext = objectives_dict.get(x.__name__)
            indiv.extra.append(float(row[ext]))
    
    indiv.creation_mode = row['creation_mode']
    all_indivs.append(indiv)

print("Calculating pareto optimal ones...")

for indiv in all_indivs:                       #Now we calculate all the individuals non dominated by any other (pareto front)
    for other_indiv in all_indivs:
        if other_indiv.dominates(indiv):
            indiv.domination_count += 1                        #Indiv is dominated by the second
    if indiv.domination_count == 0:                            #Could be done easily more efficiently, but could be interesting 
        new_pareto_indivs.append(indiv)

for p in new_pareto_indivs:                #We select individuals from the files corresponding to the pareto front ones (we filter by id)
    curr_id = p.id                      #BUT IF THERE ARE MORE THAN 1 INDIVIDUAL WITH THE SAME ID THEY WILL ALL BE ADDED, EVEN THOUGHT ONLY 1 OF THEM IS A PARETO OPTIMAL SOLUTION
    found = False
    for index, row in pareto_df.iterrows():
        if row['id'] == curr_id:
            pareto_optimal_df.append(pd.DataFrame({x : row[x] for x in pareto_df.columns.tolist()}, index=[0]))
            found = True
    if not found:
        new_pareto_indivs.remove(p)
#We extract them to a file
pareto_optimal_df = pd.concat(pareto_optimal_df)
pareto_optimal_df.to_csv('../results/general_pareto_fronts/pareto_front_' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv', index = False, header = True, columns = list(pareto_df.keys()) )

print("Execution succesful!\n------------------------------")