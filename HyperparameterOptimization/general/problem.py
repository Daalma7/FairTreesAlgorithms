import random
import string
from collections import OrderedDict as od
import sys
import copy
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append("..")
from general.individual import IndividualDT, IndividualDTGrea, IndividualLR, IndividualLRGrea, IndividualFDT, IndividualFLGBM, IndividualFLGBMGrea, IndividualFDTGrea
from general.population import Population
from general.ml import decode, print_properties_lgbm, num_leaves, train_model, evaluate_fairness, val_model, save_model, test_model, gmean_inv, fpr_diff, ppv_diff, pnr_diff, data_weight_avg_depth, print_properties_tree, get_max_depth_FLGBM


PATH_TO_DATA = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/datasets/data/'
PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/results/'

class Problem:
    """
    Class that respresents a multiobjective optimization problem
    """
    def __init__(self, objectives, extra, num_of_variables, variables_range, individuals_df, num_of_generations, num_of_individuals, dataset_name, variable_name, model, seed, expand=True, same_range=False):
        """
        Class constructor
            Parameters:
                - objectives: List containing strings of the names of the objective functions to optimize
                - extra: List containing strings of the names of the extra objective functions to measure, but not to optimize
                - num_of_variables: Number of hyperparmaeters
                - variables_range: Range that these hyperparameters can take
                - individuals_df: Dataframe where to store information about individuals obtained
                - num_of_generations: Number of generations
                - num_of_individuals: Number of individuals
                - dataset_name: Name of the dataset
                - variable_name: Name of the protected attribute
                - model: Name of the algorithm used for base classifiers
                - seed: Random seed
                - expand: Auxiliary variable, if True, methods related to training and saving will work (indicates if try to "expand" the solutions of a problem or let them stay the same as they were)
                - same_range: Says if all the hyperparameters have the same range (default=False)
        """
        
        self.num_of_objectives = len(objectives)        # Number of objectives to minimize
        self.num_of_variables = num_of_variables        # Num of variables that represent a solution for our problem
        self.objectives = objectives                    # objective *functions* to minimize
        self.extra = extra                              # objective *functions* to be calculated but for which we're not going to optimize
        self.expand = expand                            # Boolean. if True, functions are considered as f(x,y,z), and if False, as f([x,y,z])
        self.variables_range = []                       # Range of variables that represent a solution for our problem
        self.individuals_df = individuals_df            # Individuals DataFrame        
        self.num_of_generations = num_of_generations    # Number of generations to do
        self.num_of_individuals = num_of_individuals    # Number of individuals in a population
        self.dataset_name = dataset_name                # Dataset name
        self.variable_name = variable_name              # Name of the sensitive variable
        self.model = model                              # Model to learn
        self.seed = seed                                # Random seed for partition
        x_train, y_train, x_val, y_val, x_test, y_test = self.read_datasets()
        self.x_train = x_train                          # Training dataset
        self.y_train = y_train                          # Training attribute to predict
        self.x_val = x_val                              # Validation dataset
        self.y_val = y_val                              # Validation attribute to predict
        self.x_test = x_test                            # Test dataset
        self.y_test = y_test                            # Test attribute to predict
        #Random seed
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range





    def read_datasets(self):
        """
        Read training, validation and tests datasets, and extracting the attribute to predict
            Returns:
                - x_train, y_train, x_val, y_val, x_test, y_test: Training, validation and test sets, as well as their attributes to predict
        """
        train = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{self.dataset_name}/{self.dataset_name}_{self.variable_name}_train_seed_{self.seed}.csv", index_col = False)
        x_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]
        val = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{self.dataset_name}/{self.dataset_name}_{self.variable_name}_val_seed_{self.seed}.csv", index_col = False)
        x_val = val.iloc[:, :-1]
        y_val = val.iloc[:, -1]
        test = pd.read_csv(f"{PATH_TO_DATA}train_val_test_standard/{self.dataset_name}/{self.dataset_name}_{self.variable_name}_test_seed_{self.seed}.csv", index_col = False)
        x_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]
        return x_train, y_train, x_val, y_val, x_test, y_test





    def get_obj_string(self):
        """
        Get string describing the objectives
            Returns:
                - string: String describing the objectives, separeted by '__'
        """
        string = self.objectives[0].__name__
        for i in range(1, len(self.objectives)):
            string += "__" + self.objectives[i].__name__
        return string
    




    def get_extra_string(self):
        """
        Get string describing the extra objectives
            Returns:
                - string: String describing the extra objectives, separeted by '__'
        """
        if self.extra == None:
            return ''
        else:
            string = '_ext_' + self.extra[0].__name__
            for i in range(1, len(self.extra)):
                string += "__" + self.extra[i].__name__
            return string





    #Generates a default decision tree using gini criteria, and with low limitations on the rest of variables
    #It generates the biggest tree as possible, as it is unbounded in those control variables
    def generate_default_individual_dt(self, kind = 'base', criterion = 'gini'):
        """
        Generate a default Decision Tree individual using gini criterion
            Parameters:
                - kind: Kind of individual, base for any algorithm, or grea for GrEA Algorithm
                - criterion: Criterion (Either gini or entropy)
            Returns:
                - individual: Created individual
        """
        assert criterion in ['gini', 'entropy'] and kind in ['base', 'grea']
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        if criterion == 'gini':
            individual.features = [0, None, 2, None, 10]
        else:
            individual.features = [1, None, 2, None, 10]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "DT", **individual.features)
        individual.creation_mode = "initialization"
        return individual





    def generate_default_individual_fdt(self, kind = 'base', criterion = 'gini'):
        """
        Generate a default Fair Decision Tree individual using gini criterion
            Parameters:
                - kind: Kind of individual, base for any algorithm, or grea for GrEA Algorithm
                - criterion: Criterion (Either gini or entropy)
            Returns:
                - individual: Created individual
        """
        assert criterion in ['gini', 'entropy'] and kind in ['base', 'grea']
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        if criterion == 'gini':
            individual.features = [0, None, 2, None, 10, None]
        else:
            individual.features = [1, None, 2, None, 10, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FDT", **individual.features)
        individual.creation_mode = "initialization"
        return individual





    def generate_default_individual_lr(self, kind = 'base', num = 'first'):
        """
        Generate first default Logistic Regression individual using entropy criterion
            Parameters:
                - kind: Kind of individual, base for any algorithm, or grea for GrEA Algorithm
                - num: Indicates if the individuals is the first or second
            Returns:
                - individual: Created individual
        """
        assert kind in ['base', 'grea'] and num in ['first', 'second']
        if kind == 'base':
            individual = IndividualLR()
        if kind == 'grea':
            individual = IndividualLRGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        if num == 'first':
            individual.features = [100, 0.0001, 1, 0, 10]
        else:
            individual.features = [100, 0.0001, 1, 1, 10]
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "LR", **individual.features)
        individual.creation_mode = "initialization"
        return individual
    




    def generate_default_individual_flgbm(self, kind = 'base', num = 'first'):
        """
        Generate first default Fair LGBM individual using entropy criterion
            Parameters:
                - kind: Kind of individual, base for any algorithm, or grea for GrEA Algorithm
                - num: Indicates if the individuals is the first or second
            Returns:
                - individual: Created individual
        """
        assert kind in ['base', 'grea'] and num in ['first', 'second']
        if kind == 'base':
            individual = IndividualFLGBM()
        if kind == 'grea':
            individual = IndividualFLGBMGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        if num == 'first':
            individual.features = [31, 20, None, 1, 100, self.x_train.shape[1], 10, 0]
        else:
            individual.features = [31, 20, None, 10, 100, self.x_train.shape[1], 10, 0]
        hyperparameters = ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'class_weight', 'fair_param']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FLGBM", **individual.features)
        individual.creation_mode = "initialization"
        return individual
    




    def generate_individual(self, kind = 'base'):
        """
        Generates a new individual
            Parameters:
                - kind: base for any algorith, or grea for GrEA Algorithm
            Returns:
                - individual: Created individual
        """
        if self.model == "DT":
            if kind == 'base':
                individual = IndividualDT()
            if kind == 'grea':
                individual = IndividualDTGrea()
            hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        if self.model == "FDT":
            if kind == 'base':
                individual = IndividualFDT()
            if kind == 'grea':
                individual = IndividualFDTGrea()
            hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
        if self.model == "LR":
            if kind == 'base':
                individual = IndividualLR()
            if kind == 'grea':
                individual = IndividualLRGrea()
            hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        if self.model == "FLGBM":
            if kind == 'base':
                individual = IndividualFLGBM()
            if kind == 'grea':
                individual = IndividualFLGBMGrea()
            hyperparameters = ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'class_weight', 'fair_param']
        
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [random.uniform(*x) for x in self.variables_range]
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, self.model, **individual.features)
        individual.creation_mode = "initialization"
        return individual





    #Validates each individual and exports its parameters and other measures to csv
    def calculate_objectives(self, individual, first_individual=False, calc_test=False):
        """
        Calculates objectives (validation) for the given individual
            Parameters:
                - individual: Individual to which calculate the objectives
                - first_individual: Boolean value to indicate whether the individual is the first or not
                - calc_test: Indicates whether to calculate test results or not for that individual. It will also recalculate the rest of objective functions anyway
        """
        if self.expand and ((not individual.calc_objectives) or calc_test):
            individual.calc_objectives=True
            hyperparameters = individual.features
            learner = train_model(self.x_train, self.y_train, self.variable_name, self.seed, self.model, self.x_val, self.y_val, **hyperparameters) #Model training
            pred = val_model(self.x_val, learner)          #Model validation

            y_fair = evaluate_fairness(self.x_val, self.y_val, pred, self.variable_name)        #For getting objectives using validation data
            individual.objectives = []

            for x in self.objectives:
                if x.__name__ == 'gmean_inv':
                    individual.objectives.append(gmean_inv(self.y_val, pred))
                elif x.__name__ == 'fpr_diff':
                    individual.objectives.append(fpr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'ppv_diff':
                    individual.objectives.append(ppv_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'pnr_diff':
                    individual.objectives.append(pnr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'num_leaves':
                    individual.objectives.append(num_leaves(learner))
                elif x.__name__ == 'data_weight_avg_depth':
                    individual.objectives.append(data_weight_avg_depth(learner))

            individual.extra = []
            if self.extra != None:
                for x in self.extra:
                    if x.__name__ == 'gmean_inv':
                        individual.extra.append(gmean_inv(y, pred))
                    elif x.__name__ == 'fpr_diff':
                        individual.extra.append(fpr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'ppv_diff':
                        individual.extra.append(ppv_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'pnr_diff':
                        individual.extra.append(pnr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'num_leaves':
                        individual.extra.append(num_leaves(learner))
                    elif x.__name__ == 'data_weight_avg_depth':
                        individual.extra.append(data_weight_avg_depth(learner))
            
            #In case we're using decision trees, as some objectives aren't initially upper bounded, we're going to use the bound defined by the values
            #of the first individual, which is unrestricted and for that reason will have the biggest possible size
            if first_individual and (self.model == "DT" or self.model == "FDT"):
                depth, leaves, _, _ = print_properties_tree(learner)      #Size attributes for Decision Tree individuals
                var_range_list = list(self.variables_range)
                var_range_list[1] = (self.variables_range[1][0], depth)
                var_range_list[3] = (self.variables_range[3][0], leaves)
                self.variable_range = []
                self.variables_range = tuple(var_range_list)
                
            elif first_individual and (self.model == "FLGBM"):
                var_range_list = list(self.variables_range)
                var_range_list[2] = (self.variables_range[2][0], get_max_depth_FLGBM(self.x_train, self.y_train, self.variable_name, self.seed, **hyperparameters)) #Model training
                var_range_list[5] = (1, float(self.x_train.shape[1]))
                self.variable_range = []
                self.variables_range = tuple(var_range_list)
            
            if self.model == "DT" or self.model == "FDT":          #Depending on the model we will have different sets of hyperparameters for that model
                depth, leaves, data_avg_depth, depth_unbalance = print_properties_tree(learner)      #Size attributes for Decision Tree individuals
                individual.actual_leaves = leaves
                individual.actual_depth = depth
                individual.actual_data_avg_depth = data_avg_depth
                individual.actual_depth_unbalance = depth_unbalance
            elif self.model == "FLGBM":          #Depending on the model we will have different sets of hyperparameters for that model
                n_estimators, n_features, feature_importance_std = print_properties_lgbm(learner)      #Size attributes for Decision Tree individuals
                individual.actual_n_estimators = n_estimators
                individual.actual_n_features = n_features
                individual.actual_feature_importance_std = feature_importance_std
            
            if calc_test:
                pred = test_model(self.x_test, learner)       #Model test (not validation as above)
                y_fair = evaluate_fairness(self.x_test, self.y_test, pred, self.variable_name)      #For getting objectives, using test data
                objectives_test = []
                for x in self.objectives:
                    if x.__name__ == 'gmean_inv':
                        objectives_test.append(gmean_inv(self.y_test, pred))
                    elif x.__name__ == 'fpr_diff':
                        objectives_test.append(fpr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'ppv_diff':
                        objectives_test.append(ppv_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'pnr_diff':
                        objectives_test.append(pnr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'num_leaves':
                        objectives_test.append(num_leaves(individual.model))
                    elif x.__name__ == 'data_weight_avg_depth':
                        objectives_test.append(data_weight_avg_depth(individual.model))
                individual.objectives_test = copy.deepcopy(objectives_test)

                if self.extra != None:
                    extra_test = []
                    for x in self.extra:
                        if x.__name__ == 'gmean_inv':
                            extra_test.append(gmean_inv(self.y_test, pred))
                        elif x.__name__ == 'fpr_diff':
                            extra_test.append(fpr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                        elif x.__name__ == 'ppv_diff':
                            extra_test.append(ppv_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                        elif x.__name__ == 'pnr_diff':
                            extra_test.append(pnr_diff(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                        elif x.__name__ == 'num_leaves':
                            extra_test.append(num_leaves(individual.model))
                        elif x.__name__ == 'data_weight_avg_depth':
                            extra_test.append(data_weight_avg_depth(individual.model))
                        
                    individual.extra_test = copy.deepcopy(extra_test)




    #Evaluates and exports to csv the pareto-optimal individuals obtained during all the execution
    def test_and_save(self, individual, first, seed, method):
        """
        Calculates test objectives for the given individual, and stores it into the pareto_individuals/runs results folder
            Parameters:
                - individual: Individual for which to calculate the test objectives and store
                - first: Boolean value indicating whether is the first individual or not
                - seed: Random partition seed
                - method: Algorithm which was used
        """
        if self.expand:

            dict_extra = None
            dict_extra_test = None

            if self.extra != None:
                dict_extra= {f"{self.extra[i].__name__}_val": individual.extra[i] for i in range(len(self.extra))}
                dict_extra_test = {f"{self.extra[i].__name__}_test": individual.extra_test[i] for i in range(len(self.extra))}

            #Dictionaries definitions to create the dataframe representing all needed data from an individual and the execution
            dict_general_info = {'ID': individual.id, 'seed': seed, 'creation_mode':individual.creation_mode}
            dict_objectives= {f"{self.objectives[i].__name__}_val": individual.objectives[i] for i in range(self.num_of_objectives)}
            dict_test = {f"{self.objectives[i].__name__}_test": individual.objectives_test[i] for i in range(self.num_of_objectives)}

            indiv_list = list(individual.features.items())
            if self.model == "DT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                dict_actual_dimensions = {'depth': individual.actual_depth, 'leaves': individual.actual_leaves, 'data_avg_depth': individual.actual_data_avg_depth, 'depth_unbalance': individual.actual_depth_unbalance}       #It's really instersting in case of DT to have this size measures
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_actual_dimensions, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_actual_dimensions, **dict_hyperparameters}

            if self.model == "FDT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight, fair_param = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'fair_param': [fair_param]}
                dict_actual_dimensions = {'leaves': individual.actual_leaves, 'depth': individual.actual_depth, 'data_avg_depth': individual.actual_data_avg_depth, 'depth_unbalance': individual.actual_depth_unbalance}       #It's really instersting in case of DT to have this size measures
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_actual_dimensions, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_actual_dimensions, **dict_hyperparameters}

            if self.model == "LR":
                max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_hyperparameters}

            if self.model == "FLGBM":
                num_leaves, min_data_in_leaf, max_depth, learning_rate, n_estimators, feature_fraction, class_weight, fair_param= [item[1] for item in indiv_list]
                dict_hyperparameters= {'num_leaves' : [num_leaves], 'min_data_in_leaf':[min_data_in_leaf], 'max_depth':[max_depth], 'learning_rate': [learning_rate], 'n_estimators': [n_estimators], 'feature_fraction': [feature_fraction], 'class_weight': [class_weight], 'fair_param': [fair_param]}
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_hyperparameters}
            
            individuals_aux = pd.DataFrame(dict_dataframe)
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            if (first):
                individuals_aux.to_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}.csv", index = False, header = True, columns = list(dict_dataframe.keys()))
            else:
                individuals_aux.to_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}.csv", index = False, mode='a', header=False, columns = list(dict_dataframe.keys()))
    


    #Calculate file with the general pareto front using all pareto fronts in every execution
    def correct_pareto(self, seed, run, method):
        """
        Method to create files for storing the Pareto-optimal individuals non dominated by their test results
            Parameters:
                - seed: Random seed
                - run: Run for which to calculate those individual
                - method: Algorithm which was used
        """
        all_indivs = []
        pareto_optimal = []
        #ATTENTION!!! As we could want to compute the hypervolume, and for returning a structure independent from the measures we use, we should NORMALIZE HERE
        objectives_results_dict = {'gmean_inv': 'gmean_inv_test', 'fpr_diff': 'fpr_diff_test', 'ppv_diff': 'ppv_diff_test', 'pnr_diff': 'pnr_diff_test'}
        objectives_results_norm_dict = {'num_leaves': 'num_leaves_test', 'data_weight_avg_depth': 'data_weight_avg_depth_test'}
        pareto_fronts = []
        read = pd.read_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed + run}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}.csv")
        pareto_fronts.append(read)


        hyperparameters = []
        pareto_fronts = pd.concat(pareto_fronts)                            #Union of all pareto fronts got in each run
        pareto_fronts.reset_index(drop=True, inplace=True)                  #Reset index because for each run all rows have repeated ones
        for index, row in pareto_fronts.iterrows():                         #We create an individual object associated with each row
            if self.model == "DT":
                indiv = IndividualDT()
                hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
                indiv.actual_leaves = row['leaves']
                indiv.actual_depth = row['depth']
                indiv.actual_data_avg_depth = row['data_avg_depth']
                indiv.actual_depth_unbalance = row['depth_unbalance']
            if self.model == "FDT":
                indiv = IndividualDT()
                hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
                indiv.actual_leaves = row['leaves']
                indiv.actual_depth = row['depth']
                indiv.actual_data_avg_depth = row['data_avg_depth']
                indiv.actual_depth_unbalance = row['depth_unbalance']
            if self.model == "LR":
                indiv = IndividualLR()
                hyperparameters = ['max_iter','tol', 'lambda', 'l1_ratio', 'class_weight']
            if self.model == "FLGBM":
                indiv = IndividualFLGBM()
                hyperparameters = ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'class_weight', 'fair_param']
            indiv.features = [row[x] for x in hyperparameters]
            indiv.id = row['ID']
            indiv.domination_count = 0
            indiv.features = od(zip(hyperparameters, indiv.features))
            indiv.objectives = []
            for x in self.objectives:
                # We will insert all objectives, normalizing every objective that should be
                obj = objectives_results_dict.get(x.__name__, "None")
                if not obj == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                    indiv.objectives.append(float(row[obj]))
                else:                                   #In other case
                    obj = objectives_results_norm_dict.get(x.__name__)
                    indiv.objectives.append(float(row[obj]) / pareto_fronts[obj].max())
            #The same with extra objectives
            indiv.extra = []
            if not self.extra == None: 
                for x in self.extra:
                    # We will insert all objectives, normalizing every objective that should be
                    ext = objectives_results_dict.get(x.__name__, "None")
                    if not ext == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                        indiv.extra.append(float(row[ext]))
                    else:                                   #In other case
                        ext = objectives_results_norm_dict.get(x.__name__)
                        indiv.extra.append(float(row[ext]) / pareto_fronts[ext].max())
            indiv.creation_mode = row['creation_mode']
            all_indivs.append(indiv)
        for indiv in all_indivs:                       #Now we calculate all the individuals non dominated by any other (pareto front)
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
        pareto_optimal_df = pd.concat(pareto_optimal_df)
        pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed + run}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}_optimal_test.csv", index = False, header = True, columns = list(pareto_fronts.keys()))
        return pd.read_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed + run}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}_optimal_test.csv")
        


    #Calculate file with the general pareto front using all pareto fronts in every execution
    def calculate_pareto_optimal(self, seed, runs, method, correct=True):
        """
        Calculates pareto optimal individuals from the optimization process.
        ATTENTION: It supooses that all runs have been executed, so the pareto optimal individulas will be
        calculated using test objectives (The model has been trained, we now select)
            Parameters:
                - seed: Random partition seed
                - runs: Number of runs which were run
                - method: Algorithm used
                - correct: If True, apply correct_pareto method, if False it won't
            Returns:
                - pareto_optimal: List containing all pareto optimal individuals
                - pareto_optimal_df: Dataframe containing all pareto optimal individuals
        """
        if self.expand:
            pareto_fronts = []
            all_indivs = []
            pareto_optimal = []
            #ATTENTION!!! As we could want to compute the hypervolume, and for returning a structure independent from the measures we use, we should NORMALIZE HERE
            objectives_results_dict = {'gmean_inv': 'gmean_inv_test', 'fpr_diff': 'fpr_diff_test', 'ppv_diff': 'ppv_diff_test', 'pnr_diff': 'pnr_diff_test'}
            objectives_results_norm_dict = {'num_leaves': 'num_leaves_test', 'data_weight_avg_depth': 'data_weight_avg_depth_test'}

            for i in range(runs):
                if correct:
                    read = self.correct_pareto(seed, i, method)
                else:
                    read = pd.read_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/runs/{self.dataset_name}/{self.dataset_name}_seed_{seed + i}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}.csv")
                pareto_fronts.append(read)

            hyperparameters = []
            pareto_fronts = pd.concat(pareto_fronts)                            #Union of all pareto fronts got in each run
            pareto_fronts.reset_index(drop=True, inplace=True)                  #Reset index because for each run all rows have repeated ones
            for index, row in pareto_fronts.iterrows():                         #We create an individual object associated with each row
                if self.model == "DT":
                    indiv = IndividualDT()
                    hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
                    indiv.actual_leaves = row['leaves']
                    indiv.actual_depth = row['depth']
                    indiv.actual_data_avg_depth = row['data_avg_depth']
                    indiv.actual_depth_unbalance = row['depth_unbalance']
                if self.model == "FDT":
                    indiv = IndividualDT()
                    hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
                    indiv.actual_leaves = row['leaves']
                    indiv.actual_depth = row['depth']
                    indiv.actual_data_avg_depth = row['data_avg_depth']
                    indiv.actual_depth_unbalance = row['depth_unbalance']
                if self.model == "LR":
                    indiv = IndividualLR()
                    hyperparameters = ['max_iter','tol', 'lambda', 'l1_ratio', 'class_weight']
                if self.model == "FLGBM":
                    indiv = IndividualFLGBM()
                    hyperparameters = ['num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction', 'class_weight', 'fair_param']
                indiv.features = [row[x] for x in hyperparameters]
                indiv.id = row['ID']
                indiv.domination_count = 0
                indiv.features = od(zip(hyperparameters, indiv.features))
                indiv.objectives = []
                for x in self.objectives:
                    # We will insert all objectives, normalizing every objective that should be
                    obj = objectives_results_dict.get(x.__name__, "None")
                    if not obj == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                        indiv.objectives.append(float(row[obj]))
                    else:                                   #In other case
                        obj = objectives_results_norm_dict.get(x.__name__)
                        indiv.objectives.append(float(row[obj]) / pareto_fronts[obj].max())
                #The same with extra objectives
                indiv.extra = []
                if not self.extra == None: 
                    for x in self.extra:
                        # We will insert all objectives, normalizing every objective that should be
                        ext = objectives_results_dict.get(x.__name__, "None")
                        if not ext == "None":                   #The objective doesn't need to be normalized to the range [0,1]
                            indiv.extra.append(float(row[ext]))
                        else:                                   #In other case
                            ext = objectives_results_norm_dict.get(x.__name__)
                            indiv.extra.append(float(row[ext]) / pareto_fronts[ext].max())
                indiv.creation_mode = row['creation_mode']
                all_indivs.append(indiv)
            for indiv in all_indivs:                       #Now we calculate all the individuals non dominated by any other (pareto front)
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
            pareto_optimal_df.to_csv(f"{PATH_TO_RESULTS}{self.model}/{method}/pareto_individuals/overall/{self.dataset_name}/{self.dataset_name}_seed_{seed}_var_{self.variable_name}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.model}_obj_{self.get_obj_string()}{self.get_extra_string()}.csv", index = False, header = True, columns = list(pareto_fronts.keys()))

            return pareto_optimal, pareto_optimal_df                   #Population of pareto front individuals