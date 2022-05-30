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
import seaborn as sns
warnings.filterwarnings("ignore")

sys.path.append("..")
from general.qualitymeasures import *
from general.ml import *
from general.individual import *
from general.population import *


alg = dat = var = obj = mod = extra = False        #Possible parameters given

#Dictionary to propperly create individuals given the objectives
general_objectives_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val', 'num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth', 'depth': 'actual_depth'}
general_objectives_tst_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst', 'num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth', 'depth': 'actual_depth'}
objectives_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val'}
objectives_dict_norm = {'num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth'}
quality_measures = ['Mean solutions', 'Proportion', 'Hypervolume', 'Spacing', 'Maximum spread', 'Overall PF spread',  'Error ratio', 'GD', 'Inverted GD']
ml_measures = ['Min', 'Q1', 'Q2', 'Q3', 'Max']

#Help function, that will let us obtain an individual list based on information in an especifical dataframe
def get_individual_list(df, model):
    df_indivs = []
    for index, row in df.iterrows():                         #We create an individual object associated with each row
        indiv = []
        if model == "DT":
            indiv = IndividualDT()
            hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
            indiv.actual_depth = row['actual_depth']
            indiv.actual_leaves = row['actual_leaves']
        if model == "LR":
            indiv = IndividualLR()
            hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        indiv.id = row['id']
        indiv.domination_count = 0
        indiv.features = [row[x] for x in hyperparameters]
        indiv.features = od(zip(hyperparameters, indiv.features))
        indiv.objectives = []
        for x in objectives:
            if not objectives_dict.get(x.__name__) == None:
                obj = objectives_dict.get(x.__name__)
                indiv.objectives.append(float(row[obj]))
            else:
                obj = objectives_dict_norm.get(x.__name__)
                indiv.objectives.append(float(row[obj]) / max_dict.get(x.__name__))
        
        indiv.creation_mode = row['creation_mode']
        df_indivs.append(indiv)
    return df_indivs


message = "\nScript for calculating quality measures for the results relative to the parameters given"
message += "\nThe following parameters have been given by the user:"

error = False
for i in range(1, len(sys.argv)):           #We're going to read all parameters
    valid = False

    message += "\n- " + sys.argv[i]
    param = sys.argv[i].split('=')
    
    if not valid and param[0] == 'alg':               #Algorithms to use
        alg = valid = True
        algorithm = param[1].split(',')

    if not valid and param[0] == 'dat':               #Database name
        dat = valid = True
        dataset = param[1]
    
    if not valid and param[0] == 'var':               #Sensitive attribute name
        var = valid = True
        variable = param[1]

    if not valid and param[0] == 'model':
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
        print('\nThis file contains the code to calculate relevant quality measures for the pareto optimal files corresponding to the parameters given. Possible parameters are:\n\n\
\t- alg=(comm separated list of algorithms): Algorithms from which to take results. Possible algorithms are nsga2, smsemoa and grea. The default is nsga2,smsemoa,grea\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- var=(variable): Name of the sensitive variable for the dataset variable. Sensitive considered variables for each of the previous datasets are: adult-race, german-age, propublica_recidivism-race, propublica_violent_recidivism-race, ricci-Race. The default is the first variable of a the dataset (It is absolutely recommendable to change)\n\n\
\t- model=(model abbreviation): Model to use. Possible models are Decision Tree (DT) and Logistic Regression (LR). The default is DT.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,dem_fpr. IMPORTANT: Objectives should be written in the same order as they were written at the fairness.py execution sentence.\n\n\
\t- extra=(comm separated list of objectives): List of objectives to not be used in optimization, but to be calculated in individuals generated. Possible values are the same as in obj. The default is None. IMPORTANT: Objectives should be written in the same order as they were written at the fairness.py execution sentence.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython calculatemeasures.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves\n\n\
Results are saved into the corresponding results/measures folder.\n\n')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print('Some of the name of the parameters introduced is invalid.\n\
Please check out for mistakes there. Possible parameters are:\n\
alg, dat, var, obj\n')
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
    extraobj = []
    print("- extra=None")
print('---')

#String describing objectives and row creation for dataset
str_obj = objectives[0].__name__
for i in range(1, len(objectives)):
    str_obj = str_obj + '__' + str(objectives[i].__name__)

if not extra:
    str_extra = ''
else:
    str_extra = '_ext_' + extraobj[0].__name__
    for i in range(1, len(extraobj)):
        str_extra = str_extra + '__' + str(extraobj[i].__name__)




#Creation of the measures dataframe
measures_df = pd.DataFrame({'Quality Measures': quality_measures, 'NSGA-II': np.full(len(quality_measures), -47).tolist(), 'SMS-EMOA': np.full(len(quality_measures), -47).tolist(), 'GrEA': np.full(len(quality_measures), -47).tolist(), 'General PF': np.full(len(quality_measures), -47).tolist()})
real_measures_dict = {general_objectives_dict.get(objectives[i].__name__): np.full(len(ml_measures), -47).tolist() for i in range(len(objectives))}
real_measures_tst_dict = {general_objectives_tst_dict.get(objectives[i].__name__): np.full(len(ml_measures), -47).tolist() for i in range(len(objectives))}
if not extra:
    extra_measures_dict = {}
    extra_measures_tst_dict = {}
else:
    extra_measures_dict = {general_objectives_dict.get(extraobj[i].__name__): np.full(len(ml_measures), -47).tolist() for i in range(len(extraobj))}
    extra_measures_tst_dict = {general_objectives_tst_dict.get(extraobj[i].__name__): np.full(len(ml_measures), -47).tolist() for i in range(len(extraobj))}
init_dict = {'Measures': ml_measures}
measures_df_2 = []
for i in range(len(algorithm) + 1):
    measures_df_2.append(pd.DataFrame({**init_dict, **real_measures_dict, **real_measures_tst_dict, **extra_measures_dict, **extra_measures_tst_dict}))


# Now we will find all files that contain pareto fronts due to the execution of an algorithm considering the parameters
if not extra:
    regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".csv")
    regex2 = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".csv")
else:
    regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".*" + str_extra + ".csv")
    regex2 = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".*" + str_extra + ".csv")


pareto_files = []           #Files where pareto optimal individuals have been stored for each execution of the algorithm with same parameters as provided
method = []
run_mean_number = []
i = 0
j = 0

#File location
for x in algorithm:
    rootdir = "../results/" + x + "/individuals"
    run_mean_number.append([])
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                pareto_files.append(rootdir + "/" + file)
                if x == 'nsga2':
                    method.append('NSGA-II')
                elif x == 'smsemoa':
                    method.append('SMS-EMOA')
                elif x == 'grea':
                    method.append('GrEA')
            if regex2.match(file):  # We will read the number of solutions for calculating the mean.
                run_mean_number[i].append(pd.read_csv(rootdir + "/" + file).shape[0])
                j+=1
    run_mean_number[i] = float(sum(run_mean_number[i])) / len(run_mean_number[i])
    i+=1
    j = 0 

                


#Let's read all the files containing a pareto front
pareto_df = []
for x in pareto_files:
    pareto_df.append(pd.read_csv(x))

#We're going to calculate the maximum value among all individuals for the objectives whose max value is not 1
max_dict = {}
total_df = pd.concat(pareto_df).drop_duplicates(subset=['id'], keep = 'first')
for x in objectives:
    if not objectives_dict_norm.get(x.__name__) == None:
        max_dict[x.__name__] = total_df[objectives_dict_norm.get(x.__name__)].max()

#Reference point (worst possible objective value in all coordinates)
ref = []
for i in range(len(objectives)):
    ref.append(1)

#########################################
# Measures for the general pareto front #
#########################################

print("-> Calculating reference using general pareto front given in " + '../results/general_pareto_fronts/pareto_front_' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv' + '\n')
#Lets create an array of individuals, being each one, an individual belonging to the general pareto front
pareto_optimal = pd.read_csv('../results/general_pareto_fronts/pareto_front_' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv' )
pareto_optimal_noreps = pareto_optimal.drop_duplicates([general_objectives_dict.get(x.__name__) for x in objectives])
pareto_optimal_indivs = get_individual_list(pareto_optimal, model)

print("\tSaving\n")

algoprop = algorithm_proportion(pareto_optimal)         #Extraction of values in the form [alg, prop, alg2, prop2, ...]
algoprop = [x for y in algoprop.split("\n") for x in y.split(" ")]
algoprop = [x for x in algoprop if x != '']
hypervol = hypervolume(pareto_optimal_indivs, ref)
spac = spacing(pareto_optimal_indivs)
mspre = maximum_spread(pareto_optimal_indivs)

for i in range(len(algoprop)):
    if algoprop[i] == 'smsemoa':
        measures_df.loc[measures_df['Quality Measures'] == 'Proportion', ['SMS-EMOA']] = algoprop[i+1]
    if algoprop[i] == 'nsga2':
        measures_df.loc[measures_df['Quality Measures'] == 'Proportion', ['NSGA-II']] = algoprop[i+1]
    if algoprop[i] == 'grea':
        measures_df.loc[measures_df['Quality Measures'] == 'Proportion', ['GrEA']] = algoprop[i+1]
measures_df.loc[measures_df['Quality Measures'] == 'Hypervolume', ['General PF']] = round(hypervol, 6)
measures_df.loc[measures_df['Quality Measures'] == 'Spacing', ['General PF']] = round(spac, 6)
measures_df.loc[measures_df['Quality Measures'] == 'Maximum spread', ['General PF']] = round(mspre, 6)

for x in objectives + extraobj:
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Min', [general_objectives_dict.get(x.__name__)]] = pareto_optimal[general_objectives_dict.get(x.__name__)].min()
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q1', [general_objectives_dict.get(x.__name__)]] = pareto_optimal[general_objectives_dict.get(x.__name__)].quantile(0.25)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q2', [general_objectives_dict.get(x.__name__)]] = pareto_optimal[general_objectives_dict.get(x.__name__)].quantile(0.5)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q3', [general_objectives_dict.get(x.__name__)]] = pareto_optimal[general_objectives_dict.get(x.__name__)].quantile(0.75)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Max', [general_objectives_dict.get(x.__name__)]] = pareto_optimal[general_objectives_dict.get(x.__name__)].max()
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Min', [general_objectives_tst_dict.get(x.__name__)]] = pareto_optimal[general_objectives_tst_dict.get(x.__name__)].min()
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q1', [general_objectives_tst_dict.get(x.__name__)]] = pareto_optimal[general_objectives_tst_dict.get(x.__name__)].quantile(0.25)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q2', [general_objectives_tst_dict.get(x.__name__)]] = pareto_optimal[general_objectives_tst_dict.get(x.__name__)].quantile(0.5)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Q3', [general_objectives_tst_dict.get(x.__name__)]] = pareto_optimal[general_objectives_tst_dict.get(x.__name__)].quantile(0.75)
    measures_df_2[-1].loc[measures_df_2[-1]['Measures'] == 'Max', [general_objectives_tst_dict.get(x.__name__)]] = pareto_optimal[general_objectives_tst_dict.get(x.__name__)].max()


##############################################
# Measures for each algorithm's pareto front #
##############################################

pareto_optimal_indivs = get_individual_list(pareto_optimal, model)
for i in range(len(pareto_df)):

    print("-> Reading data from file: " + pareto_files[i])

    # Getting all individuals
    indivs = get_individual_list(pareto_df[i], model)
    
    #Let's extract features of the execution employed by the file name
    divided_name = pareto_files[i].split("_")

    for j in range(len(divided_name)):
        if divided_name[j] == "baseseed":
            seed = divided_name[j+1]
        if divided_name[j] == "nruns":
            runs = divided_name[j+1]
        if divided_name[j] == "gen":
            generations = divided_name[j+1]
        if divided_name[j] == "indiv":
            individuals = divided_name[j+1]
    
    print("\t Saving\n")

    hypervol = hypervolume(indivs, ref)
    spac = spacing(indivs)
    mspre = maximum_spread(indivs)
    opfs = overall_pareto_front_spread(indivs, pareto_optimal_indivs)
    er = error_ratio(indivs, pareto_optimal_indivs)
    gd = generational_distance(indivs, pareto_optimal_indivs)
    igd = inverted_generational_distance(indivs, pareto_optimal_indivs)

    measures_df.loc[measures_df['Quality Measures'] == 'Mean solutions', [method[i]]] = run_mean_number[i]
    measures_df.loc[measures_df['Quality Measures'] == 'Hypervolume', [method[i]]] = hypervol
    measures_df.loc[measures_df['Quality Measures'] == 'Spacing', [method[i]]] = spac
    measures_df.loc[measures_df['Quality Measures'] == 'Maximum spread', [method[i]]] = mspre
    measures_df.loc[measures_df['Quality Measures'] == 'Overall PF spread', [method[i]]] = opfs
    measures_df.loc[measures_df['Quality Measures'] == 'Error ratio', [method[i]]] = er
    measures_df.loc[measures_df['Quality Measures'] == 'GD', [method[i]]] = gd
    measures_df.loc[measures_df['Quality Measures'] == 'Inverted GD', [method[i]]] = igd

    for x in objectives + extraobj:
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Min', [general_objectives_dict.get(x.__name__)]] = pareto_df[i][general_objectives_dict.get(x.__name__)].min()
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q1', [general_objectives_dict.get(x.__name__)]] = pareto_df[i][general_objectives_dict.get(x.__name__)].quantile(0.25)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q2', [general_objectives_dict.get(x.__name__)]] = pareto_df[i][general_objectives_dict.get(x.__name__)].quantile(0.5)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q3', [general_objectives_dict.get(x.__name__)]] = pareto_df[i][general_objectives_dict.get(x.__name__)].quantile(0.75)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Max', [general_objectives_dict.get(x.__name__)]] = pareto_df[i][general_objectives_dict.get(x.__name__)].max()
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Min', [general_objectives_tst_dict.get(x.__name__)]] = pareto_df[i][general_objectives_tst_dict.get(x.__name__)].min()
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q1', [general_objectives_tst_dict.get(x.__name__)]] = pareto_df[i][general_objectives_tst_dict.get(x.__name__)].quantile(0.25)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q2', [general_objectives_tst_dict.get(x.__name__)]] = pareto_df[i][general_objectives_tst_dict.get(x.__name__)].quantile(0.5)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Q3', [general_objectives_tst_dict.get(x.__name__)]] = pareto_df[i][general_objectives_tst_dict.get(x.__name__)].quantile(0.75)
        measures_df_2[i].loc[measures_df_2[i]['Measures'] == 'Max', [general_objectives_tst_dict.get(x.__name__)]] = pareto_df[i][general_objectives_tst_dict.get(x.__name__)].max()

measures_df['General PF'] = measures_df['General PF'].astype(str)
measures_df['General PF'] = measures_df['General PF'].replace(str(-47.0), '-')

measures_df.to_csv('../results/measures/qmeasures_' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv', index = False, header = True, columns = list(measures_df.keys()))
#print(measures_df.to_latex(index=False))

for i in range(len(measures_df_2)-1):
    measures_df_2[i].to_csv('../results/measures/ml_' + dataset + '_var_' + variable + '_alg_' + algorithm[i] + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv', index = False, header = True, columns = list(measures_df_2[i].keys()))
measures_df_2[-1].to_csv('../results/measures/mlgeneral_' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.csv', index = False, header = True, columns = list(measures_df_2[-1].keys()))

print("Results saved at "+ '../results/measures/' + dataset + '_var_' + variable + '_model_' + model + '_obj_' + str(str_obj) + str(str_extra) + '.txt')

##### COMMENT ALL BELOW FOR EXECUTING ON YOUR OWN
#Median values

#We read files of the 2d pareto fronts found
regex = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_DT.*_obj_gmean_inv__dem_fpr" + ".*_ext_.*.csv")
rootdir = "../results/nsga2/individuals"
files2d = []
medianindivs = []

i = 0
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            files2d.append(pd.read_csv(rootdir + "/" + file))

#Then we select the individual at the median of error_val (if there are more than 1, we average them)
files2d = pd.concat(files2d)
indiv = files2d.loc[files2d['error_val']==files2d['error_val'].median()]
if indiv.shape[0] == 0: #If the median just falls between 2 values, we find the nearest
    ln_ind = files2d.loc[files2d['error_val']<files2d['error_val'].median()]['error_val'].idxmax()
    un_ind = files2d.loc[files2d['error_val']>files2d['error_val'].median()]['error_val'].idxmin()
    medianindivs.append(files2d.iloc[[ln_ind, un_ind], 1:].mean().to_frame().T)    #.T.mean().to_frame().T
    #nearest = files2d[i].iloc[files2d[i]['error_val'].sub(files2d[i]['error_val'].median()).abs().idxmin(), 1:].to_frame().T.mean().to_frame().T)
    #upper = files2d[i].iloc[files2d[i]['error_val'].sub(files2d[i]['error_val'].median()).abs().idxmin(), 1:].to_frame().T.mean().to_frame().T)
else:
    medianindivs.append(indiv.mean().to_frame().T)
i+=1

median = medianindivs[0].iloc[0,:]['error_val']

#Now we will select files from the 4 objectives optimization

regex = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_DT.*_obj_gmean_inv__dem_fpr__dem_ppv.*.csv")

for x in ['nsga2', 'smsemoa', 'grea']:
    curindivs = []
    rootdir = "../results/" + x + "/individuals"
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                df = pd.read_csv(rootdir + "/" + file)
                indiv = df.loc[df['error_val'] == median]
                lower = df.iloc[df.loc[df['error_val']< median]['error_val'].sort_values()[-2:].index, 1:]
                upper = df.iloc[df.loc[df['error_val']> median]['error_val'].sort_values()[:2].index, 1:]
                near = pd.concat([lower, upper])                    #We join them
                near['weight'] = np.abs(near['error_val']-median)   #Calculate their distance to the median
                near['weight'] = near['weight']/near['weight'].min()    #Times bigger than the min distance
                near['weight'] = 1/((1/near['weight']).sum()) / near['weight']      #Calculate the actual weights
                #And now we calculate the actual weighted average.
                near = near.drop(columns=['creation_mode'])
                near = near.mul(near['weight'].to_numpy(), axis=0).sum().to_frame().T       #Individual resulting from the weighted average
                curindivs.append(near.iloc[:, :-1])
    longi = len(curindivs)
    curindivs = pd.concat(curindivs)
    medianindivs.append(curindivs.sum().to_frame().T / longi)

medianindivs = pd.concat(medianindivs)
medianindivs.insert(0, "alg", ['nsga2_2', 'nsga2_4', 'smsemoa', 'grea'], True)

medianindivs.to_csv('../results/measures/median_' + dataset + '_var_' + variable + '_model_' + model + '.csv', index = False, header = True, columns = list(medianindivs.keys()))

#Coverage analysis

all_indivs = []
new_pareto_indivs = []
pareto_optimal_df = []

regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".*" + str_extra + ".csv")
rootdir = "../results/nsga2/individuals"
extra_df = []
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            extra_df = pd.read_csv(rootdir + "/" + file)
extra_df.insert(loc=1, column='algorithm', value=np.full(extra_df.shape[0], 'nsga2_2').tolist())

pareto_df = pd.concat([pareto_optimal, extra_df]) #Union of all dataframes

#Before continuing we need to get rid of duplicates using the id column, as the next code cant deal with them
pareto_df = pareto_df.drop_duplicates(subset=['id'], keep = 'first')

# Getting all individuals 
for index, row in pareto_df.iterrows():                         #We create an individual object associated with each row
    if model == "DT":
        indiv = IndividualDT()
        indiv.features = [float(row['criterion']), row['max_depth'], row['min_samples_split'], row['max_leaf_nodes'], row['class_weight']]
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
        obj = general_objectives_dict.get(x.__name__)
        indiv.objectives.append(float(row[obj]))
    
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



if(len(objectives) == 4):
    result = np.zeros([4,4])
    algoprop = algorithm_proportion(pareto_optimal_df)
    algoprop = algoprop.split()
    for i in range(len(algoprop)):
        if algoprop[i] == 'nsga2_2':
            result[0,0] = float(algoprop[i+1]) 
        if algoprop[i] == 'nsga2':
            result[1,1] = float(algoprop[i+1]) 
        if algoprop[i] == 'smsemoa':
            result[2,2] = float(algoprop[i+1]) 
        if algoprop[i] == 'grea':
            result[3,3] = float(algoprop[i+1]) 

    regex = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".csv")
    regex2 = re.compile("individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_.*_ext_.*.csv")

    dict4obj = {'nsga2':[], 'smsemoa':[], 'grea':[]}
    df2obj = []
    for x in ['nsga2', 'smsemoa', 'grea']:
        rootdir = "../results/" + x + "/individuals"
        for root, dirs, files in os.walk(rootdir):
            for file in files:
                if regex.match(file):
                    dict4obj[x].append(pd.read_csv(rootdir + "/" + file))
                if regex2.match(file):  # We will read the number of solutions for calculating the mean.
                    df2obj.append(pd.read_csv(rootdir + "/" + file))
    
    dict4obj = {x: get_individual_list(pd.concat(dict4obj.get(x)).drop_duplicates([general_objectives_dict.get(x.__name__) for x in objectives]), model) for x in ['nsga2', 'smsemoa', 'grea']}
    df2obj = get_individual_list(pd.concat(df2obj).drop_duplicates([general_objectives_dict.get(x.__name__) for x in objectives]), model)
    
    result[0,1] = coverage(df2obj, dict4obj['nsga2'])
    result[0,2] = coverage(df2obj, dict4obj['smsemoa'])
    result[0,3] = coverage(df2obj, dict4obj['grea'])
    result[1,0] = coverage(dict4obj['nsga2'], df2obj)
    result[1,2] = coverage(dict4obj['nsga2'], dict4obj['smsemoa'])
    result[1,3] = coverage(dict4obj['nsga2'], dict4obj['grea'])
    result[2,0] = coverage(dict4obj['smsemoa'], df2obj)
    result[2,1] = coverage(dict4obj['smsemoa'], dict4obj['nsga2'])
    result[2,3] = coverage(dict4obj['smsemoa'], dict4obj['grea'])
    result[3,0] = coverage(dict4obj['grea'], df2obj)
    result[3,1] = coverage(dict4obj['grea'], dict4obj['nsga2'])
    result[3,2] = coverage(dict4obj['grea'], dict4obj['smsemoa'])

    resultdict = {}
    resultdict['nsga2_2'] = result[:,0]
    resultdict['nsga2_4'] = result[:,1]
    resultdict['smsemoa'] = result[:,2]
    resultdict['grea'] = result[:,3]
    df = pd.DataFrame(resultdict)
    df = df.rename(index={0:'NSGA-II₂', 1:'NSGA-II₄', 2:'SMS-EMOA', 3:'GrEA'})
    df = df.rename(columns={'nsga2_2':'NSGA-II₂', 'nsga2_4':'NSGA-II₄', 'smsemoa':'SMS-EMOA', 'grea':'GrEA'})
    sns.heatmap(df, annot=True, cmap='Blues', fmt=".4f", annot_kws={'size':16})
    plt.xlabel("Covered")
    plt.ylabel("Covering")
    path = "../results/images/" + dataset + "_var_" + variable + '_model_' + model + "_obj_" + str_obj
    plt.savefig(path + "/coverage.png", dpi=200)
    df.insert(0, "-", ['NSGA-II₂', 'NSGA-II₄', 'SMS-EMOA', 'GrEA'], True)
    df.to_csv('../results/measures/covermatrix' + dataset + '_var_' + variable + '_model_' + model + '.csv', index = False, header = True, columns = list(df.keys()))

    plt.clf()
    # Creation of coverage related graphics
    dat = pd.DataFrame({'NSGA-II₂':[0.4436], 'NSGA-II₄':[0.0797], 'SMS-EMOA':[0.4516], 'GrEA':[0.0247]})
    sns.barplot(data=dat, ci=None)
    plt.savefig("../results/images/nondominated.png", dpi=200)
    plt.clf()
    dat2 = pd.DataFrame({'NSGA-II₂':[7], 'NSGA-II₄':[5], 'SMS-EMOA':[12], 'GrEA':[0]})
    sns.barplot(data=dat2, ci=None)
    plt.savefig("../results/images/rankingcover.png", dpi=200)


print("Execution succesful!\n------------------------------")