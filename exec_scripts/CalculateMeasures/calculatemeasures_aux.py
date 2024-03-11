import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from qualitymeasures import hypervolume, spacing, maximum_spread, error_ratio, overall_pareto_front_spread, generational_distance, inverted_generational_distance, ideal_point, nadir_point, algorithm_proportion, diff_val_test_rate, coverage

PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/results'
datasetlist = ['academic','adult','arrhythmia','bank','catalunya','compas','credit','crime','default','diabetes-w','diabetes','drugs','dutch','german','heart','hrs','insurance','kdd-census','lsat','nursery','obesity', 'older-adults','oulad','parkinson','ricci','singles','student','tic','wine','synthetic-athlete','synthetic-disease','toy']
dict_outcomes = {'academic': 'atd','adult': 'income','arrhythmia': 'arrhythmia','bank': 'Subscribed','catalunya': 'recid','compas': 'score','credit': 'NoDefault','crime': 'ViolentCrimesPerPop','default': 'default','diabetes-w': 'Outcome','diabetes': 'readmitted','drugs': 'Coke','dutch': 'status','german': 'Label','heart': 'class','hrs': 'score','insurance': 'charges','kdd-census': 'Label','lsat':'ugpa','nursery': 'class','obesity': 'NObeyesdad','older-adults': 'mistakes','oulad': 'Grade','parkinson': 'total_UPDRS','ricci': 'Combine','singles': 'income','student': 'G3','tic': 'income', 'wine': 'quality','synthetic-athlete': 'Label','synthetic-disease': 'Label','toy': 'Label'}
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}


palette = {'DT': '#1f77b4', 'FDT': '#ff7f0e', 'GP': '#2ca02c', 'FLGBM': '9467bd', 'Total':'d62728'}
sns.set_style("darkgrid")

class Individual(object):
    """
    Individual class, which will serve us to calculate Pareto optimal individuals
    """

    def __init__(self):
        self.id = None
        self.algorithm = None
        self.creation_mode = None
        self.objectives = None
        self.objectives_test = None
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    # Prev: dominates_standard
    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives_test, other_individual.objectives_test):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)


def get_str(obj, extra):
    obj_str = '__'.join(obj)
    extra_str = ''
    if not extra is None:
        extra_str = '__'.join(extra)
        extra_str = '_ext_' + extra_str
    return obj_str, extra_str


def create_results_df():
    """
    Creates dataframe with result values
        Parameters:
            -
        Returns:
            -
    """
    #Creation of the measures dataframe
    measures_df_1 = pd.DataFrame({'Quality Measures': quality_measures, 'NSGA-II': np.full(len(quality_measures), -47).tolist(), 'SMS-EMOA': np.full(len(quality_measures), -47).tolist(), 'GrEA': np.full(len(quality_measures), -47).tolist(), 'General PF': np.full(len(quality_measures), -47).tolist()})
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

    return measures_df_1, measures_df_2





def read_overall_pareto_files(dataname, models, bseed, nind, ngen, obj, extra):
    """
    Read all individuals of the overall pareto files, for each model considered
        Parameters:
            - dataname: Dataset name to read the files
            - models: Execution models from which to read (FDT, FLGBM, GP...)
            - bseed: Base seed (only for file name)
            - nind: Number of individuals (only for file name)
            - ngen: Number of generations (only for file name)
            - obj: Objectives to consider
            - extra: Extra objectives, for which the algorithm will not optimize
        Returns:
            - indivs: List containing all individuals
    """

    obj_str, extra_str = get_str(obj, extra)
    
    indivs = []

    dict_plot = {obj[i]: [] for i in range(len(obj))}
    dict_plot['algorithm'] = []
    dict_plot_test = {obj[i]: [] for i in range(len(obj))}
    dict_plot_test['algorithm'] = []

    for model in models:
        new_indiv_list = []
        if model in ['DT', 'FDT', 'FLGBM']:
            prev_path = f"{PATH_TO_RESULTS}/{model}/nsga2/"
        else:
            prev_path = f"{PATH_TO_RESULTS}/{model}"

        pareto_optimal_df = pd.read_csv(f"{prev_path}/pareto_individuals/overall/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{nind}_model_{model}_obj_{obj_str}{extra_str}.csv")

        exclude = ['ID', 'creation_mode']
        for x in obj:
            exclude.append(x+'_val')
            exclude.append(x+'_test')
        for index, row in pareto_optimal_df.iterrows():                         #We create an individual object associated with each row
            indiv = Individual()
            indiv.id = row['ID']
            indiv.algorithm = model
            indiv.creation_mode = row['creation_mode']
            indiv.objectives = [row[x+'_val'] for x in obj]
            indiv.objectives_test = [row[x+'_test'] for x in obj]
            indiv.rank = None
            indiv.crowding_distance = None
            indiv.domination_count = 0
            indiv.dominated_solutions = 0
            indiv.features = {a: row[a] for a in pareto_optimal_df.columns if not a in exclude}
            new_indiv_list.append(indiv)

            [dict_plot[obj[i]].append(indiv.objectives[i]) for i in range(len(obj))]
            dict_plot['algorithm'].append(model)
            [dict_plot_test[obj[i]].append(indiv.objectives_test[i]) for i in range(len(obj))]
            dict_plot_test['algorithm'].append(model)
        
        indivs.append(new_indiv_list)


    plt.title(f"{dataname} individuals' validation objectives")
    sns.scatterplot(pd.DataFrame(dict_plot), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_val_{dataname}")
    plt.close()

    plt.title(f"{dataname} individuals' test objectives")
    sns.scatterplot(pd.DataFrame(dict_plot_test), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_test_{dataname}")
    plt.close()

    return indivs




def create_total_pareto_optimal(df_indivs, dataname, models, bseed, nind, ngen, obj, extra):
    """
    Creates the pareto_optimal individuals using all information available of all algorithms.
    It also stores the information in memory.
        Parameters:
            - indivs: List containing all individuals from the overall pareto files for the rest of the parameters
            - dataname: Dataset name to read the files
            - models: Execution models from which to read (FDT, FLGBM, GP...)
            - bseed: Base seed (only for file name)
            - nind: Number of individuals (only for file name)
            - ngen: Number of generations (only for file name)
            - obj: Objectives to consider
            - extra: Extra objectives, for which the algorithm will not optimize
        Returns:
            - pareto_optimal: List containing all pareto-optimal individuals
    """

    obj_str, extra_str = get_str(obj, extra)

    columns = {'ID': [], 'algorithm':[], 'creation_mode':[]}
    for o in obj:
        columns[f"{o}_val"] = []
    for o in obj:
        columns[f"{o}_test"] = []

    all_features = []

    # Calculation of optimal individuals
    pareto_optimal = []
    for individual in df_indivs:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in df_indivs:
            if individual.dominates(other_individual):                  # If the current individual dominates the other
                individual.dominated_solutions.append(other_individual) # It is added to its list of dominated solutions
            elif other_individual.dominates(individual):                # If the other dominates the current
                individual.domination_count += 1                        # We add 1 to its domination count

    for individual in df_indivs:
        stop = 0
        if individual.domination_count == 0:                            # If any solution dominates it
            pareto_optimal.append(individual)
            for feat in individual.features:
                columns[feat] = []
                all_features.append(feat)
    
    all_features = set(all_features)

    # Pareto optimal individuals calculated
    # Creating now the store dataset 
    dict_plot = {obj[i]: [] for i in range(len(obj))}
    dict_plot['algorithm'] = []
    dict_plot_test = {obj[i]: [] for i in range(len(obj))}
    dict_plot_test['algorithm'] = []

    for individual in pareto_optimal:
        columns['ID'].append(individual.id)
        columns['algorithm'].append(individual.algorithm)
        columns['creation_mode'].append(individual.creation_mode)

        for i in range(len(obj)):
            columns[f"{obj[i]}_val"].append(individual.objectives[i])
            columns[f"{obj[i]}_test"].append(individual.objectives_test[i])
        
        for feat in all_features:
            if feat in individual.features:
                columns[feat].append(individual.features[feat])
            else:
                columns[feat].append("-")

        [dict_plot[obj[i]].append(individual.objectives[i]) for i in range(len(obj))]
        dict_plot['algorithm'].append(individual.algorithm)
        [dict_plot_test[obj[i]].append(individual.objectives_test[i]) for i in range(len(obj))]
        dict_plot_test['algorithm'].append(individual.algorithm)

    store_df = pd.DataFrame(data=columns)
    store_df.to_csv(f"{PATH_TO_RESULTS}/ParetoOptimal/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{nind}_obj_{obj_str}{extra_str}.csv", index=False)

    plt.title(f"{dataname} individuals' validation objectives")
    sns.scatterplot(pd.DataFrame(dict_plot), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_po_val_{dataname}")
    plt.close()

    plt.title(f"{dataname} individuals' test objectives")
    sns.scatterplot(pd.DataFrame(dict_plot_test), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_po_test_{dataname}")
    plt.close()
    return pareto_optimal
    



def drop_duplicates():
    """
    Remove individuals with duplicated parameters, if any
        Parameters:
            - indivlist: list containing Individuals
        Returns:
            - indivnodup: same list without repeated individuals 
    """
    pass





def calculate_quality_metrics():
    """
    Calculate quality metrics for every
        Parameters:
            -
        Returns:
            -
    """
    pass





def calculate_general_pareto_front_measures(pareto_optimal, dataname, obj, extra):
    """
    Calculate quality metrics and all other kinds of metrics for the general pareto front
        Parameters:
            - pareto_optimal: list containing all pareto optimal individuals
            - dataname: dataset name
            - obj: objective list
            - extra: extra objectives list
        Returns:
            - results: measured metrics
    """
    #Lets create an array of individuals, being each one, an individual belonging to the general pareto front
    #pareto_optimal_noreps = pareto_optimal.drop_duplicates([general_objectives_dict.get(x.__name__) for x in objectives])

    results = {}
    results['Proportion'] = algorithm_proportion(pareto_optimal) # Algorithm proportion
    results['Hypervolume'] = hypervolume(pareto_optimal)          # Hypervolume
    results['Spacing'] = spacing(pareto_optimal)                  # Spacing
    results['Maximum Spread'] = maximum_spread(pareto_optimal)          # Maximum spread
    objectives_dict = {}
    for i in range(len(obj)):
        obj_list = np.array([indiv.objectives_test[i] for indiv in pareto_optimal])
        objectives_dict[f"{obj[i]}_min"] = obj_list.min()
        objectives_dict[f"{obj[i]}_q1"] = np.quantile(obj_list, 0.25)
        objectives_dict[f"{obj[i]}_q2"] = np.quantile(obj_list, 0.5)
        objectives_dict[f"{obj[i]}_q3"] = np.quantile(obj_list, 0.75)
        objectives_dict[f"{obj[i]}_max"] = obj_list.max()
        objectives_dict[f"{obj[i]}_std"] = obj_list.std()

    # This is for displaying the actual values instead of percentages
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_format
    plt.pie(results['Proportion'].values(), labels = results['Proportion'].keys(), autopct=autopct_format(results['Proportion'].values()), colors=[palette[x] for x in results['Proportion'].keys()])
    plt.title("Pie plot showing proportion of each algorithm")
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/pie_proportion_{dataname}")
    plt.close()
    return results





def calculate_algorithm_pareto_front_measures(indivs, pareto_optimal, obj, extra):
    """
    Calculate quality metrics and all other kinds of metrics for the overall pareto front of a given algorithm
        Parameters:
            - indivs: list of the pareto optimal individuals to which calculate the metrics
        Returns:
            - results: Measured metrics
    """
    results = {}
    results['Hypervolume'] = hypervolume(indivs)                                      # Hypervolume
    results['Spacing'] = spacing(indivs)                                              # Spacing
    results['Maximum Spread'] = maximum_spread(indivs)                                      # Maximum spread
    results['Overall Pareto Front Spread'] = overall_pareto_front_spread(indivs, pareto_optimal)          # Overall Pareto fron spread
    results['Error ratio'] = error_ratio(indivs, pareto_optimal)                            # Error ratio
    results['Generational Distance'] = generational_distance(indivs, pareto_optimal)                  # Generational distance
    results['Inverted Generational Distance'] = inverted_generational_distance(indivs, pareto_optimal)        # Inverted generational distance


    objectives_dict = {}
    for i in range(len(obj)):
        obj_list = np.array([indiv.objectives_test[i] for indiv in pareto_optimal])
        objectives_dict[f"{obj[i]}_min"] = obj_list.min()
        objectives_dict[f"{obj[i]}_q1"] = np.quantile(obj_list, 0.25)
        objectives_dict[f"{obj[i]}_q2"] = np.quantile(obj_list, 0.5)
        objectives_dict[f"{obj[i]}_q3"] = np.quantile(obj_list, 0.75)
        objectives_dict[f"{obj[i]}_max"] = obj_list.max()
        objectives_dict[f"{obj[i]}_std"] = obj_list.std()

    return results


"""
def calculate_median_values():
"""
"""
    Calculate median values for the solutions of all datasets
        Parameters:
            -
        Returns:
            -
"""
"""
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
"""





def plot_algorithm_metrics(results, po_results, models, dataname):
    """
    Plot results metrics for each dataset
        Parameters:
            - results: Results dictionary
            - models: All considered models
    """
    highlow_dict = {'Hypervolume': 'Higher',
                    'Spacing': 'Higher',
                    'Maximum Spread': 'Higher',
                    'Overall Pareto Front Spread': 'Higher',
                    'Error ratio': 'Lower',
                    'Generational Distance': 'Lower',
                    'Inverted Generational Distance': 'Lower'
                    }
    
    # Add results for the Pareto optimal set
    results_with_po = {}    
    for elem in po_results:
        if not elem == 'Proportion':
            results_with_po[elem] = results[elem]
            results_with_po[elem].append(po_results[elem])
            del results[elem]

    # Create dataframes and store
    results = pd.DataFrame(data=results, index=models)
    results['models'] = results.index
    results_with_po = pd.DataFrame(data=results_with_po, index=models + ['Total'])
    results_with_po['models'] = results_with_po.index

    for data in [results, results_with_po]:
        for elem in data.columns:
            if not elem == 'models':
                sns.barplot(data, x='models', y=elem, hue='models')
                plt.title(f"Barplot showing {elem} for {dataname} dataset. ({highlow_dict[elem]} is better)")
                plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/barplot_{elem}_{dataname}.png", dpi=200)
                plt.close()
    
    





def coverage_analysis(algorithms, indiv_lists, dataname):
    """
    Performs a coverage analysis over the pareto front individuals belonging to the given populations
        Parameters:
            - algorithms: Different algorithms employed
            - indiv_list: Lists of individual object which are the solutions for each algorithm
            - dataname: Name of the dataset
            - models
    """

    all_indivs = []
    new_pareto_indivs = []
    pareto_optimal_df = []

    # TODO: Drop duplicate individuals

    df_algorithms = {alg: [] for alg in algorithms}

    for i in range(len(algorithms)):
        for j in range(len(algorithms)):
            df_algorithms[algorithms[j]].append(coverage(indiv_lists[i], indiv_lists[j]))
    
    df = pd.DataFrame(df_algorithms, index=algorithms)
    print(df)
    sns.heatmap(df, annot=True, cmap='Blues', fmt=".4f", annot_kws={'size':16})
    plt.xlabel("Covered")
    plt.ylabel("Covering")
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/coverage_{dataname}.png", dpi=200)
    plt.close()