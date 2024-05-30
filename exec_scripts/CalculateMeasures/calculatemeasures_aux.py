import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

import numpy as np
from scipy.interpolate import PchipInterpolator
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from qualitymeasures import hypervolume, spacing, maximum_spread, error_ratio, overall_pareto_front_spread, generational_distance, inverted_generational_distance, ideal_point, nadir_point, algorithm_proportion, div_test_val_rate, coverage

PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/results'
datasetlist = ['academic','adult','arrhythmia','bank','catalunya','compas','credit','crime','default','diabetes-w','diabetes','drugs','dutch','german','heart','hrs','insurance','kdd-census','lsat','nursery','obesity', 'older-adults','oulad','parkinson','ricci','singles','student','tic','wine','synthetic-athlete','synthetic-disease','toy']
dict_outcomes = {'academic': 'atd','adult': 'income','arrhythmia': 'arrhythmia','bank': 'Subscribed','catalunya': 'recid','compas': 'score','credit': 'NoDefault','crime': 'ViolentCrimesPerPop','default': 'default','diabetes-w': 'Outcome','diabetes': 'readmitted','drugs': 'Coke','dutch': 'status','german': 'Label','heart': 'class','hrs': 'score','insurance': 'charges','kdd-census': 'Label','lsat':'ugpa','nursery': 'class','obesity': 'NObeyesdad','older-adults': 'mistakes','oulad': 'Grade','parkinson': 'total_UPDRS','ricci': 'Combine','singles': 'income','student': 'G3','tic': 'income', 'wine': 'quality','synthetic-athlete': 'Label','synthetic-disease': 'Label','toy': 'Label'}
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}


hyperparams_alg= {
    'DT': ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight'],
    'FDT': ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param'],
    'FGP': ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance'],
    'FLGBM': ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'fair_param']
}
colors_hyperparams = {'criterion': '#f44336',           # Red
                      'max_depth': '#e81e63',           # Pink
                      'min_samples_split': '#9c27b0',   # Purple
                      'max_leaf_nodes': '#673ab7',      # Deep Purple
                      'class_weight': '#3f51b5',        # Indigo 
                      'fair_param': '#2196f3',          # Blue
                      'prunings': '#03a9f4',            # Light Blue
                      'leaves': '#00bcd4',              # Cyan
                      'depth': '#009688',               # Teal
                      'data_avg_depth': '#4caf50',      # Green
                      'depth_unbalance': '#8bc34a',     # Light Green
                      'num_leaves': '#cddc39',          # Lime
                      'min_data_in_leaf': '#ffeb3b',    # Yellow
                      'learning_rate': '#ffc107',       # Amber
                      'n_estimators': '#ff9800',        # Orange
                      'feature_fraction': '#ff5722'     # Deep Orange
                      }

palette = {'DT': '#1f77b4', 'DT-mean': '#0e3653',
           'FDT': '#ff7f0e', 'FDT-mean': '#8f4300',
           'FGP': '#2ca02c', 'FGP-mean': '#185818',
           'FLGBM': '#9467bd', 'FLGBM-mean': '#553375',
           'Total':'#d62728'}
palette_obj = {'gmean_inv': '#a1c9f4', 'fpr_diff': '#8de5a1'}

sns.set_theme(style='darkgrid', palette='bright')
sns.set_style('darkgrid')

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


"""
def create_results_df():

    Creates dataframe with result values. Not currently used

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
"""




def read_runs_pareto_files(dataname, models, bseed, nind, ngen, obj, extra, n_runs=10):
    """
    Read all individuals of the overall pareto files, for each model considered
        Parameters:
            - dataname: Dataset name to read the files
            - models: Execution models from which to read (FDT, FLGBM, FGP...)
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
    indivs_part = []

    dict_plot = {obj[i]: [] for i in range(len(obj))}
    dict_plot['algorithm'] = []
    dict_plot_test = {obj[i]: [] for i in range(len(obj))}
    dict_plot_test['algorithm'] = []

    for model in models:
        indivs_part = []
        if model in ['DT', 'FDT', 'FLGBM']:
            prev_path = f"{PATH_TO_RESULTS}/{model}/nsga2/"
        else:
            prev_path = f"{PATH_TO_RESULTS}/{model}"
        for run in range(n_runs):
            new_indiv_list = []
            pareto_optimal_df = pd.read_csv(f"{prev_path}/pareto_individuals/runs/{dataname}/{dataname}_seed_{bseed + run}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{nind}_model_{model}_obj_{obj_str}{extra_str}.csv")

            exclude = ['ID', 'creation_mode']
            for x in obj:
                exclude.append(x+'_val')
                exclude.append(x+'_test')
            seen_test_objs = []                                                     # We do not want repeated individuals
            for index, row in pareto_optimal_df.iterrows():                         # We create an individual object associated with each row
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
                if not indiv.objectives_test in seen_test_objs:
                    new_indiv_list.append(indiv)
                    seen_test_objs.append(indiv.objectives_test)

                [dict_plot[obj[i]].append(indiv.objectives[i]) for i in range(len(obj))]
                dict_plot['algorithm'].append(model)
                [dict_plot_test[obj[i]].append(indiv.objectives_test[i]) for i in range(len(obj))]
                dict_plot_test['algorithm'].append(model)
            indivs_part.append(new_indiv_list)
        indivs.append([])
        for lista in indivs_part:
            for indiv in lista:
                indivs[-1].append(indiv)


    plt.title(f"Pareto optimal sets for {dataname} dataset by algorithm (validation)")
    sns.scatterplot(pd.DataFrame(dict_plot), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_val_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.title(f"Pareto optimal sets for {dataname} dataset by algorithm (test)")
    sns.scatterplot(pd.DataFrame(dict_plot_test), x=obj[0], y=obj[1], hue='algorithm', alpha=0.5, palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_test_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    return indivs



def create_total_pareto_optimal(df_indivs, dataname, bseed, nind, ngen, obj, extra, nruns=10):
    """
    Creates the pareto_optimal individuals using all information available of all algorithms.
    It also stores the information in memory.
        Parameters:
            - indivs: List containing all individuals from the overall pareto files for the rest of the parameters
            - dataname: Dataset name to read the files
            - bseed: Base seed (only for file name)
            - nind: Number of individuals (only for file name)
            - ngen: Number of generations (only for file name)
            - obj: Objectives to consider
            - extra: Extra objectives, for which the algorithm will not optimize
            - nruns: Number of runs that were executed
        Returns:
            - pareto_optimal: List containing all pareto-optimal individuals
    """

    columns = {'ID': [], 'algorithm':[], 'creation_mode':[]}
    for o in obj:
        columns[f"{o}_val"] = []
    for o in obj:
        columns[f"{o}_test"] = []


    # In the first place, we will calculate the synthetic individuals which will conform the 'Pareto Front' for each algorithm

    # We first consider, for each individual, the algorithm which generated it and its objectives values
    temp_columns = {'algorithm':[]}
    for o in obj:
        temp_columns[f"{o}"] = []
    
    for indiv in df_indivs:
        temp_columns['algorithm'].append(indiv.algorithm)
        for i, o in enumerate(obj):
            temp_columns[f"{o}"].append(indiv.objectives_test[i])

    # Then we convert it to a pandas DataFrame
    df_calculate = pd.DataFrame(data=temp_columns)
    models = df_calculate['algorithm'].unique()

    pareto_optimal_alg = []

    # Finally, we calculate the average pareto optimal set
    for model in models:
        pareto_optimal_alg.append([])
        df_model = df_calculate.loc[df_calculate['algorithm'] == model]
        num_indivs = int(np.ceil(float(df_model.shape[0]) / nruns))
        if num_indivs < 1:
            num_indivs = 1
        print(model, num_indivs)
        prev_objectives = [None, None]
        for i in range(num_indivs):
            # Ensure that do not consider repeated individuals
            if num_indivs > 1:
                cur_objectives = [df_model.loc[:, f'{obj[0]}'].quantile(float(i)/(num_indivs-1)), df_model.loc[:, f'{obj[1]}'].quantile(1.0 - float(i)/(num_indivs-1))]
            else:
                cur_objectives = [df_model.loc[:, f'{obj[0]}'].median(), df_model.loc[:, f'{obj[1]}'].median()]
            if not prev_objectives == cur_objectives:
                # Correct x values for plotting purposes.
                if not prev_objectives[0] is None and cur_objectives[0] <= prev_objectives[0]:
                    cur_objectives[0] = prev_objectives[0] + 1e-15
                if not prev_objectives[1] is None and cur_objectives[1] >= prev_objectives[1]:
                    cur_objectives[1] = prev_objectives[1] - 1e-15
                new_indiv = Individual()
                new_indiv.algorithm = model
                new_indiv.creation_mode = 'Pareto Quantile'
                new_indiv.objectives_test = cur_objectives
                prev_objectives = cur_objectives
                pareto_optimal_alg[-1].append(new_indiv)
        
        # The correction done can surpass the limits established for the objective space, reason why we need to correct them
        # First dimension correction, applying linear recalculation of range
        cur_min = next_min = pareto_optimal_alg[-1][0].objectives_test[0]
        cur_max = next_max = pareto_optimal_alg[-1][-1].objectives_test[0]
        renorm = False
        if pareto_optimal_alg[-1][0].objectives_test[0] < 0:
            next_min = 0
            renorm = True
        if pareto_optimal_alg[-1][-1].objectives_test[0] > 1:
            next_max = 1
            renorm = True
        if cur_max == cur_min:
            renorm = False
        if renorm:
            for i in range(len(pareto_optimal_alg[-1])):
                pareto_optimal_alg[-1][i].objectives_test[0] = ((next_max - next_min) / (cur_max - cur_min)) * (pareto_optimal_alg[-1][i].objectives_test[0] - cur_max) + next_max
        
        # Second dimension correction, applying linear recalculation of range
        cur_min = next_min = pareto_optimal_alg[-1][-1].objectives_test[1]
        cur_max = next_max = pareto_optimal_alg[-1][0].objectives_test[1]
        if pareto_optimal_alg[-1][0].objectives_test[1] > 1:
            next_max = 1
            renorm = True
        if pareto_optimal_alg[-1][-1].objectives_test[1] < 0:
            next_min = 0
            renorm = True
        if cur_max == cur_min:
            renorm = False
        if renorm:
            for i in range(len(pareto_optimal_alg[-1])):
                pareto_optimal_alg[-1][i].objectives_test[1] = ((next_max - next_min) / (cur_max - cur_min)) * (pareto_optimal_alg[-1][i].objectives_test[1] - cur_max) + next_max
    # After it, we will calculate the optimal individuals among them, to be the 'Total Pareto front'
    # Calculation of optimal individuals
    allindivs = [indiv for alg in pareto_optimal_alg for indiv in alg]

    pareto_optimal = []
    for individual in allindivs:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in allindivs:
            if individual.dominates(other_individual):                  # If the current individual dominates the other
                individual.dominated_solutions.append(other_individual) # It is added to its list of dominated solutions
            elif other_individual.dominates(individual):                # If the other dominates the current
                individual.domination_count += 1                        # We add 1 to its domination count

    for individual in allindivs:
        if individual.domination_count == 0:                            # If any solution dominates it
            pareto_optimal.append(individual)

    # Pareto optimal individuals calculated
    # Creating now the store dataset

    dfs_plot_alg = []

    for indiv_alg_list in pareto_optimal_alg:
        dict_plot_alg = {}
        dict_plot_alg['algorithm'] = []
        for i in range(len(obj)):
            dict_plot_alg[obj[i]] = [] 

        for individual in indiv_alg_list:
            columns['algorithm'].append(individual.algorithm)
            columns['creation_mode'].append(individual.creation_mode)

            for i in range(len(obj)):
                columns[f"{obj[i]}_test"].append(individual.objectives_test[i])
            
            dict_plot_alg['algorithm'].append(individual.algorithm)
            for i in range(len(obj)):
                dict_plot_alg[obj[i]].append(individual.objectives_test[i])
        dfs_plot_alg.append(pd.DataFrame(dict_plot_alg))
        
    
    dict_plot_total = {}
    dict_plot_total['algorithm'] = []
    for i in range(len(obj)):
        dict_plot_total[obj[i]] = [] 

    for individual in pareto_optimal:
        columns['algorithm'].append(individual.algorithm)
        columns['creation_mode'].append(individual.creation_mode)

        for i in range(len(obj)):
            columns[f"{obj[i]}_test"].append(individual.objectives_test[i])
        
        dict_plot_total['algorithm'].append(individual.algorithm)
        for i in range(len(obj)):
            dict_plot_total[obj[i]].append(individual.objectives_test[i])

    df_plot_total = pd.DataFrame(data=dict_plot_total).sort_values(by=[obj[0]]).reset_index(drop=True)
    #store_df = pd.DataFrame(data=columns)
    #store_df.to_csv(f"{PATH_TO_RESULTS}/ParetoOptimal/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{nind}_obj_{obj_str}{extra_str}.csv", index=False)


    # Plot data by algorithm
    plt.title(f"Synthetic Pareto front by algorithm for {dataname} dataset")
    sns.scatterplot(df_calculate, x=obj[0], y=obj[1], hue='algorithm', alpha=0.3, palette=palette, legend=False)

    for model, df in zip(models, dfs_plot_alg):
        x_interp = np.linspace(df[obj[0]].min(), df[obj[0]].max(), 1000)
        y_interp = PchipInterpolator(df[obj[0]], df[obj[1]], extrapolate=True)(x_interp)
        sns.lineplot(x=x_interp, y=y_interp, color=palette[model])
        sns.scatterplot(df, x=obj[0], y=obj[1], hue='algorithm', palette=palette)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_po_algorithm_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # Plot general optimal data
    plt.title(f"Synthetic general Pareto front for {dataname} dataset")
    x_interp = np.linspace(df_plot_total[obj[0]].min(), df_plot_total[obj[0]].max(), 1000)
    y_interp = PchipInterpolator(df_plot_total.drop_duplicates(obj[0])[obj[0]], df_plot_total.drop_duplicates(obj[0])[obj[1]], extrapolate=True)(x_interp)
    sns.lineplot(x=x_interp, y=y_interp, color='r')
    sns.scatterplot(df_plot_total, x=obj[0], y=obj[1], hue='algorithm', alpha=0.9, palette=palette)
    for model, df in zip(models, dfs_plot_alg):
        sns.scatterplot(df, x=obj[0], y=obj[1], hue='algorithm', alpha=0.3, palette=palette, legend=False)
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/scatter_po_total_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    return pareto_optimal_alg, pareto_optimal
    



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





def calculate_general_pareto_front_measures(pareto_optimal, dataname, obj, extra=None):
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
    plt.title(f"Proportion of individuals by algorithm in the synthetic Pareto front for {dataname} dataset")
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/pie_proportion_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    return results





def calculate_algorithm_pareto_front_measures(indivs, pareto_optimal):
    """
    Calculate quality metrics and all other kinds of metrics for the overall pareto front of a given algorithm
        Parameters:
            - indivs: list of individuals (typically pareto optimal individuals of some algorithm) which calculate the metrics
            - pareto_optimal: list of pareto optimal solutions among all algorithms and runs, the best of the best
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

    return results


def plot_div_test_val(dataname, models, indivs_list, obj, filter=True, p_iqr=5):
    """
    Plots the difference between validation and test sets
        Parameters:
            - dataname: Name of the dataset
            - models: list containing the name of the algorithms to evaluate
            - indivs_list: list containing the individuals for which to calculate
            - obj: objectives to calculate the difference
            - filter: boolean value which indicates wether to filter really extreme values or not
            - p_iqr: multiplying parameter for iqr to select individuals to filter
    """
    cumm_df = None
    for i in range(len(models)):
        if cumm_df is None:
            cumm_df = div_test_val_rate(models[i], indivs_list[i], obj)
            cumm_df['test/val'] = cumm_df['test/val'].astype(float)

        else:
            cumm_df = pd.concat([cumm_df, div_test_val_rate(models[i], indivs_list[i], obj)])
    
    # Filtering (greatly benefits the graphic)
    if filter:
        for alg in models:
            part_cumm_df = cumm_df.loc[cumm_df['algorithm'] == alg]
            iqr = part_cumm_df['test/val'].quantile(0.75) - part_cumm_df['test/val'].quantile(0.25)
            mean = part_cumm_df['test/val'].mean()
            cumm_df = cumm_df.drop(cumm_df.loc[(cumm_df['algorithm'] == 'alg') & (cumm_df['test/val'] > mean + p_iqr*iqr)].index)
            cumm_df = cumm_df.drop(cumm_df.loc[(cumm_df['algorithm'] == 'alg') & (cumm_df['test/val'] < mean - p_iqr*iqr)].index)

    sns.violinplot(cumm_df, x='algorithm', y='test/val', width=0.9, hue='measure', palette=[palette_obj[x] for x in obj], density_norm='width', log_scale=True)
    plt.title(f"Relation between test and validation (overfit) for {dataname} dataset")
    plt.ylabel('Test/Validation results')
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/violin_overfit_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()







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
    Plot comparative barplots showing quality metrics among models
        Parameters:
            - results: Results dictionary
            - po_result: 
            - models: All considered models
            - dataname: Name of the dataset
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
                sns.barplot(data, x='models', y=elem, hue='models', palette=[palette[x] for x in data['models']])
                plt.title(f"Barplot showing {elem} for {dataname} dataset\n({highlow_dict[elem]} is better)")
                plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/barplot_{elem}_{dataname}.pdf", format="pdf", bbox_inches='tight')
                plt.close()
    
    





def coverage_analysis(algorithms, indiv_lists, dataname):
    """
    Performs a coverage analysis over the pareto front individuals belonging to the given populations
        Parameters:
            - algorithms: Different algorithms employed
            - indiv_list: List of individual object which are the solutions for each algorithm
            - dataname: Name of the dataset
    """
    # TODO: Drop duplicate individuals

    df_algorithms = {alg: [] for alg in algorithms}

    for i in range(len(algorithms)):
        for j in range(len(algorithms)):
            df_algorithms[algorithms[j]].append(coverage(indiv_lists[i], indiv_lists[j]))
    
    df = pd.DataFrame(df_algorithms, index=algorithms)
    sns.heatmap(df, annot=True, cmap='Blues', vmin=0, vmax=1, fmt=".4f", annot_kws={'size':16})
    plt.title(f"Coverage analysis using synthetic Pareto fronts by algorithm for {dataname} dataset")
    plt.xlabel("Covered")
    plt.ylabel("Covering")
    plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/coverage_{dataname}.pdf", format='pdf', bbox_inches='tight')
    plt.close()



def plot_generation_stats(dataname, models, bseed, runs, nind, ngen, obj, extra):
    """
    Previous function which calls to the line_evolution function. It specifies the parameters to execute it.
    """
    obj_str, extra_str = get_str(obj, extra)

    # First step, read all dataframes containing generation stats
    for model in models:
        if model in ['DT', 'FDT', 'FLGBM']:
            prev_path = f"{PATH_TO_RESULTS}/{model}/nsga2"
        else:
            prev_path = f"{PATH_TO_RESULTS}/{model}"

        generations_dfs = []
        for i in range(runs):
            generations_dfs.append(pd.read_csv(f"{prev_path}/generation_stats/{dataname}/{dataname}_seed_{bseed + i}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{nind}_model_{model}_obj_{obj_str}{extra_str}.csv"))
        


        # Second step, create a new dataset with values being the average
        df_concat = pd.concat(generations_dfs)        # Concatenate them
        grouped = df_concat.groupby(df_concat.index)    # Group by index

        # Compute mean and standard deviation
        mean_df = grouped.mean()
        std_df = grouped.std()

        if model in ['DT', 'FDT']:
            metrics = ['leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
        elif model == 'FGP':
            metrics = ['prunings', 'leaves', 'depth', 'data_avg_depth', 'depth_unbalance']
        else:
            metrics = ['n_estimators', 'n_features', 'feature_importance_std']
        line_evolution(dataname, model, mean_df, std_df, metrics, True, False, True)
 
    
   
def line_evolution(dataname, model, mean_df, std_df, metrics, include_ext=True, plot_std=True, store=False):
    """
    Plot line plots showing evolution of different parameters through generations
        Parameters:
            - dataname: Name of the dataset
            - model: String containing the algorithm employed 
            - bseed: Base random seed
            - ngen: Number of generations
            - indiv: Number of individuals within the population
            - obj_str: String containing the objective functions
            - metrics: Metrics to take into account
            - include_ext: Include extra objectives or not 
            - store: Boolean attribute. If true, the graphic is stored. In any other case, it will be plotted on screen
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 3*len(metrics)), sharey=False)
    fig.suptitle(f"Metrics of Fair Decision Trees in each generation")
    plt.gcf().subplots_adjust(bottom=0.1)

    metric_colors = {'strong_prunings': 'darkorange', 'medium_prunings': 'orange', 'weak_prunings': 'gold',
                     'strong_leaves': 'darkgreen', 'medium_leaves': 'forestgreen', 'weak_leaves': 'lightgreen',
                     'strong_depth': 'darkred', 'medium_depth': 'red', 'weak_depth': 'lightsalmon',
                     'strong_data_avg_depth': 'darkblue', 'medium_data_avg_depth':'dodgerblue' , 'weak_data_avg_depth': 'lightblue',
                     'strong_depth_unbalance': 'darkviolet', 'medium_depth_unbalance':'violet', 'weak_depth_unbalance': 'plum',
                     'strong_n_estimators': 'darkviolet', 'medium_n_estimators': 'violet', 'weak_n_estimators': 'plum',
                     'strong_n_features': 'darkblue', 'medium_n_features': 'dodgerblue', 'weak_n_features': 'lightblue',
                     'strong_feature_importance_std': 'darkred', 'medium_feature_importance_std': 'red', 'weak_feature_importance_std': 'lightsalmon',
}

    for i, metric in enumerate(metrics):
        mean_df[f"+std"] = mean_df[f"mean_{metric}"] + mean_df[f"std_{metric}"]
        mean_df[f"-std"] = mean_df[f"mean_{metric}"] - mean_df[f"std_{metric}"]
        for pos in ['mean', 'min', 'max']: 
            mean_df[f"{pos}_{metric}+std"] = mean_df[f"{pos}_{metric}"] + std_df[f"{pos}_{metric}"]
            mean_df[f"{pos}_{metric}-std"] = mean_df[f"{pos}_{metric}"] - std_df[f"{pos}_{metric}"]

        if include_ext:
            sns.lineplot(ax = axes[i], x=range(mean_df.shape[0]), y=mean_df[f"max_{metric}"], label='max,min', color=metric_colors[f"medium_{metric}"])
            sns.lineplot(ax = axes[i], x=range(mean_df.shape[0]), y=mean_df[f"min_{metric}"], color=metric_colors[f"medium_{metric}"])
        sns.lineplot(ax = axes[i], x=range(mean_df.shape[0]), y=mean_df[f"mean_{metric}"], label = 'mean', color = metric_colors[f"strong_{metric}"])

        axes[i].fill_between(x=range(mean_df.shape[0]), y1=mean_df[f"+std"], y2=mean_df[f"-std"], color=metric_colors[f"medium_{metric}"], alpha=0.4)
        if plot_std:
            for pos in ['mean', 'min', 'max']: 
                axes[i].fill_between(x=range(mean_df.shape[0]), y1=mean_df[f"{pos}_{metric}+std"], y2=mean_df[f"{pos}_{metric}-std"], color=metric_colors[f"weak_{metric}"], alpha=0.3)

        #axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Generation')
        axes[i].set_ylabel(f'{metric}')
    
    fig.tight_layout()

    if store:
        plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/gen_evolution_{dataname}_{model}.pdf", format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()




def metrics_ranking(models, results):
    """
    Plot rankinkgs as barplots of each quality metric for all datasets
        Parameters:
            - models: list containing strings of names of algorithms which were used
            - results: dictionary containing each metric as key, and the rankings of each model as value
    """
    for elem in results:
        sns.barplot(x=models, y=results[elem], hue=models, palette=[palette[x] for x in models])
        plt.title(f"Barplot showing rakings for {elem} metric\nconsidering all datasets (The higher the better)")
        plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/rankings/ranking_{elem}.pdf", format="pdf", bbox_inches='tight')
        plt.close()



def hyperparameter_plots(algorithms, indiv_lists, dataname=None):
    """
    Plots violinplots for every attribute on a Pareto front set.
        Parameters:
            - algorithms: Different algorithms employed
            - indiv_list: List of individual object which are the solutions for each algorithm
            - dataname: Name of the dataset
    """
    for alg, ind_list in zip(algorithms, indiv_lists):
        hyperparam_df = pd.DataFrame(data={param:[ind.features[param] for ind in ind_list] for param in hyperparams_alg[alg]})
        if alg == 'FDT':
            hyperparam_df = hyperparam_df.fillna(5)
        
        # Assert there aren't any null value
        assert(hyperparam_df, hyperparam_df.isnull().sum().sum() == 0)

        correlation = hyperparam_df.corr()

        # Correlation heatmap (Corrplot)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={'size':16})
        if dataname is None:
            plt.title(f"Corrplot of hyperparameters for {alg} and all datasets")
            plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/hyperparameters/corrplot_{alg}.pdf", format="pdf", bbox_inches='tight')
            plt.close()
        else:
            plt.title(f"Corrplot of hyperparameters for {alg} and {dataname} dataset")
            plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/corrplot_{alg}_{dataname}.pdf", format="pdf", bbox_inches='tight')
            plt.close()

        # Violinplot for each hyperparameter
        for param in hyperparams_alg[alg]:
            sns.violinplot(hyperparam_df, y=param, width=0.8, color=colors_hyperparams[param])
            if dataname is None:
                plt.title(f"Violin plot of {param} for {alg} and all datasets")
                plt.ylabel(f"{param}")
                plt.xlabel("Density")
                plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/hyperparameters/violin_{alg}_{param}.pdf", format="pdf", bbox_inches='tight')
                plt.close()
            else:
                plt.title(f"Violin plot of {param} for {alg} and {dataname} dataset")
                plt.ylabel(f"{param}")
                plt.xlabel("Density")
                plt.savefig(f"{PATH_TO_RESULTS}/GeneralGraphics/{dataname}/violin_{alg}_{param}_{dataname}.pdf", format="pdf", bbox_inches='tight')
                plt.close()