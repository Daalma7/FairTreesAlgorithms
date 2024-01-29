import random
import string
from collections import OrderedDict as od
import sys

sys.path.append("..")
from general.individual import *
from general.population import Population
from general.ml import *

#Clase que representa un problema multiobjetivo
class Problem:

    def __init__(self, objectives, extra, num_of_variables, variables_range, individuals_df, num_of_generations, num_of_individuals, dataset_name, variable_name, model, seed, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)        #Number of objectives to minimize
        self.num_of_variables = num_of_variables        #Num of variables that represent a solution for our problem
        self.objectives = objectives                    #objective *functions* to minimize
        self.extra = extra                              #objective *functions* to be calculated but for which we're not going to optimize
        self.expand = expand                            #Boolean. if True, functions are considered as f(x,y,z), and if False, as f([x,y,z])
        self.variables_range = []                       #Range of variables that represent a solution for our problem
        self.individuals_df = individuals_df            #Individuals DataFrame        
        self.num_of_generations = num_of_generations    #Number of generations to do
        self.num_of_individuals = num_of_individuals    #Number of individuals in a population
        self.dataset_name = dataset_name                #Dataset name
        self.variable_name = variable_name              #Name of the sensitive variable
        self.model = model                              #Model to learn
        self.seed = seed                                #Random seed
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def get_obj_string(self):
        string = self.objectives[0].__name__
        for i in range(1, len(self.objectives)):
            string += "__" + self.objectives[i].__name__
        return string
    
    def get_extra_string(self):
        if self.extra == None:
            return ''
        else:
            string = '_ext_' + self.extra[0].__name__
            for i in range(1, len(self.extra)):
                string += "__" + self.extra[i].__name__
            return string

    #Generates a default decision tree using gini criteria, and with low limitations on the rest of variables
    #It generates the biggest tree as possible, as it is unbounded in those control variables
    def generate_default_individual_gini_dt(self, kind = 'base'):
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [0, None, 2, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "DT", **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    #Generates a default decision tree using entropy criteria, and with low limitations on the rest of variables
    #It generates the biggest tree as possible, as it is unbounded in those control variables
    def generate_default_individual_entropy_dt(self, kind = 'base'):
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [1, None, 2, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "DT", **individual.features)
        individual.creation_mode = "inicialization"
        return individual
    
    def generate_default_individual_gini_fdt(self, kind = 'base'):
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [0, None, 2, None, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FDT", **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    #Generates a default decision tree using entropy criteria, and with low limitations on the rest of variables
    #It generates the biggest tree as possible, as it is unbounded in those control variables
    def generate_default_individual_entropy_fdt(self, kind = 'base'):
        if kind == 'base':
            individual = IndividualDT()
        if kind == 'grea':
            individual = IndividualDTGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [1, None, 2, None, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FDT", **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    def generate_first_default_lr(self, kind= 'base'):
        if kind == 'base':
            individual = IndividualLR()
        if kind == 'grea':
            individual = IndividualLRGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [100, 0.0001, 1, 0, None]
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "LR", **individual.features)
        individual.creation_mode = "inicialization"
        return individual
    
    def generate_second_default_lr(self, kind= 'base'):
        if kind == 'base':
            individual = IndividualLR()
        if kind == 'grea':
            individual = IndividualLRGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [100, 0.0001, 1, 1, None]
        hyperparameters = ['max_iter', 'tol', 'lambda', 'l1_ratio', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "LR", **individual.features)
        individual.creation_mode = "inicialization"
        return individual
    
    def generate_first_default_flgbm(self, kind= 'base'):
        if kind == 'base':
            individual = IndividualFLGBM()
        if kind == 'grea':
            individual = IndividualFLGBMGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [1, 31, 20, None, 0.1, 100, 1.0, 20]
        hyperparameters = ['lamb', 'num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FLGBM", **individual.features)
        individual.creation_mode = "inicialization"
        return individual
    
    def generate_second_default_flgbm(self, kind= 'base'):
        if kind == 'base':
            individual = IndividualFLGBM()
        if kind == 'grea':
            individual = IndividualFLGBMGrea()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [1, 31, 20, None, 0.01, 100, 1.0, 20]
        hyperparameters = ['lamb', 'num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, "FLGBM", **individual.features)
        individual.creation_mode = "inicialization"
        return individual
    
    #Generates a random decision tree
    def generate_individual(self, kind = 'base'):
        if self.model == "DT":
            if kind == 'base':
                individual = IndividualDT()
            if kind == 'grea':
                individual = IndividualDTGrea()
            hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        if self.model == "FDT":
            if kind == 'base':
                individual = IndividualDT()
            if kind == 'grea':
                individual = IndividualDTGrea()
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
            hyperparameters = ['lamb', 'num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction']
        
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))          
        individual.features = [random.uniform(*x) for x in self.variables_range]
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, self.model, **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    #Validates each individual and exports its parameters and other measures to csv
    def calculate_objectives(self, individual, first_individual, seed, method):
        if self.expand:
            objectives_results_dict = {'gmean_inv': 'error', 'dem_fpr': 'dem_fp', 'dem_ppv': 'dem_ppv', 'dem_pnr': 'dem_pnr', 'num_leaves': 'num_leaves', 'data_weight_avg_depth':'data_weight_avg_depth'}
            hyperparameters = individual.features
            learner = train_model(self.dataset_name, self.variable_name, self.seed, self.model, **hyperparameters) #Model training
            X, y, pred = val_model(self.dataset_name, learner, seed)          #Model validation

            y_fair = evaluate_fairness(X, y, pred, self.variable_name)        #For getting objectives using validation data
            individual.objectives = []

            for x in self.objectives:
                if x.__name__ == 'gmean_inv':
                    individual.objectives.append(gmean_inv(y, pred))
                elif x.__name__ == 'dem_fpr':
                    individual.objectives.append(dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'dem_ppv':
                    individual.objectives.append(dem_ppv(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'dem_pnr':
                    individual.objectives.append(dem_pnr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'num_leaves':
                    individual.objectives.append(num_leaves(learner))
                elif x.__name__ == 'data_weight_avg_depth':
                    individual.objectives.append(data_weight_avg_depth(learner, X, self.seed))

            individual.extra = []
            if self.extra != None:
                for x in self.extra:
                    if x.__name__ == 'gmean_inv':
                        individual.extra.append(gmean_inv(y, pred))
                    elif x.__name__ == 'dem_fpr':
                        individual.extra.append(dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'dem_ppv':
                        individual.extra.append(dem_ppv(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'dem_pnr':
                        individual.extra.append(dem_pnr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'num_leaves':
                        individual.extra.append(num_leaves(learner))
                    elif x.__name__ == 'data_weight_avg_depth':
                        individual.extra.append(data_weight_avg_depth(learner, X, self.seed))
            #In case we're using decision trees, as some objectives aren't initially upper bounded, we're going to use the bound defined by the values
            #of the first individual, which is unrestricted and for that reason will have the biggest possible size
            if first_individual and (self.model == "DT" or self.model == "FDT"):
                depth, leaves = print_properties_tree(learner)      #Size attributes for Decision Tree individuals
                var_range_list = list(self.variables_range)
                var_range_list[1] = (self.variables_range[1][0], depth)
                var_range_list[3] = (self.variables_range[3][0], leaves)
                self.variable_range = []
                self.variables_range = tuple(var_range_list)
                
            if first_individual and (self.model == "FLGBM"):
                var_range_list = list(self.variables_range)
                var_range_list[3] = (self.variables_range[3][0], get_max_depth_FLGBM(self.dataset_name, self.variable_name, self.seed, self.model, **hyperparameters)) #Model training
                self.variable_range = []
                self.variables_range = tuple(var_range_list)

            #Dictionaries definitions to create the dataframe representing all needed data from an individual and the execution
            indiv_list = list(individual.features.items())
            if self.model == "DT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                depth, leaves = print_properties_tree(learner)      #Size attributes for Decision Tree individuals
                individual.actual_depth = depth
                individual.actual_leaves = leaves
            indiv_list = list(individual.features.items())
            if self.model == "FDT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight, fair_param = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'fair_param':[fair_param]}
                depth, leaves = print_properties_tree(learner)      #Size attributes for Decision Tree individuals
                individual.actual_depth = depth
                individual.actual_leaves = leaves
            if self.model == "LR":
                max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
            if self.model == "FLGBM":
                lamb, num_leaves, min_data_in_leaf, max_depth, learning_rate, n_estimators, feature_fraction = [item[1] for item in indiv_list]
                dict_hyperparameters= {'lamb': [lamb], 'num_leaves' : [num_leaves], 'min_data_in_leaf':[min_data_in_leaf], 'max_depth':[max_depth], 'learning_rate': [learning_rate], 'n_estimators': [n_estimators], 'feature_fraction': [feature_fraction]}
            dict_general_info = {'id': individual.id, 'creation_mode':individual.creation_mode}
            dict_objectives= {objectives_results_dict.get(self.objectives[i].__name__): individual.objectives[i] for i in range(self.num_of_objectives)}
            if self.extra != None: 
                dict_extra= {objectives_results_dict.get(self.extra[i].__name__): individual.extra[i] for i in range(len(self.extra))}
                dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_hyperparameters} #Union
            else:
                dict_dataframe = {**dict_general_info, **dict_objectives, **dict_hyperparameters} #Union

            individuals_aux = pd.DataFrame(dict_dataframe)
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            self.individuals_df.to_csv('../results/' + self.model + '/' + str(method) + '/individuals/individuals_' + self.dataset_name + '_seed_' + str(seed) + '_var_' + self.variable_name + '_gen_' + str(self.num_of_generations) + '_indiv_' + str(self.num_of_individuals) + '_model_' + self.model + '_obj_' + self.get_obj_string() + self.get_extra_string() + '.csv', index = False, header = True, columns = list(dict_dataframe.keys()))
    
    #Evaluates and exports to csv the pareto-optimal individuals obtained during all the execution
    def test_and_save(self, individual, first, seed, method):
        if self.expand:
            objectives_results_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val', 'num_leaves': 'num_leaves', 'data_weight_avg_depth':'data_weight_avg_depth'}
            objectives_test_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst', 'num_leaves': 'num_leaves_tst', 'data_weight_avg_depth':'data_weight_avg_depth_tst'}
            hyperparameters = individual.features
            learner = train_model(self.dataset_name, self.variable_name, seed, self.model, **hyperparameters)
            save_model(learner, self.dataset_name, seed, self.variable_name, self.num_of_generations, self.num_of_individuals, individual.id, self.model, method, self.objectives)
            X, y, pred = test_model(self.dataset_name, learner, seed)       #Model test (not validation as above)
            y_fair = evaluate_fairness(X, y, pred, self.variable_name)      #For getting objectives, using test data
            objectives_test = []
            for x in self.objectives:
                if x.__name__ == 'gmean_inv':
                    objectives_test.append(gmean_inv(y, pred))
                elif x.__name__ == 'dem_fpr':
                    objectives_test.append(dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'dem_ppv':
                    objectives_test.append(dem_ppv(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'dem_pnr':
                    objectives_test.append(dem_pnr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                elif x.__name__ == 'num_leaves':
                    objectives_test.append(num_leaves(learner))
                elif x.__name__ == 'data_weight_avg_depth':
                    objectives_test.append(data_weight_avg_depth(learner, X, self.seed))

            extra_test = []
            if self.extra != None:
                for x in self.extra:
                    if x.__name__ == 'gmean_inv':
                        extra_test.append(gmean_inv(y, pred))
                    elif x.__name__ == 'dem_fpr':
                        extra_test.append(dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'dem_ppv':
                        extra_test.append(dem_ppv(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'dem_pnr':
                        extra_test.append(dem_pnr(y_fair[0], y_fair[1], y_fair[2], y_fair[3]))
                    elif x.__name__ == 'num_leaves':
                        extra_test.append(num_leaves(learner))
                    elif x.__name__ == 'data_weight_avg_depth':
                        extra_test.append(data_weight_avg_depth(learner, X, self.seed))
                dict_extra= {objectives_results_dict.get(self.extra[i].__name__): individual.extra[i] for i in range(len(self.extra))}
                dict_extra_test = {objectives_test_dict.get(self.extra[i].__name__): extra_test[i] for i in range(len(self.extra))}


            #Dictionaries definitions to create the dataframe representing all needed data from an individual and the execution
            dict_general_info = {'id': individual.id, 'seed': seed, 'creation_mode':individual.creation_mode}
            dict_objectives= {objectives_results_dict.get(self.objectives[i].__name__): individual.objectives[i] for i in range(self.num_of_objectives)}
            dict_test = {objectives_test_dict.get(self.objectives[i].__name__): objectives_test[i] for i in range(self.num_of_objectives)}

            indiv_list = list(individual.features.items())
            if self.model == "DT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                dict_actual_dimensions = {'actual_depth': individual.actual_depth, 'actual_leaves': individual.actual_leaves}       #It's really instersting in case of DT to have this size measures
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_actual_dimensions, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_test, **dict_hyperparameters}

            if self.model == "FDT":          #Depending on the model we will have different sets of hyperparameters for that model
                criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight, fair_param = [item[1] for item in indiv_list]
                dict_hyperparameters= {'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'fair_param': [fair_param]}
                dict_actual_dimensions = {'actual_depth': individual.actual_depth, 'actual_leaves': individual.actual_leaves}       #It's really instersting in case of DT to have this size measures
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_actual_dimensions, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_test, **dict_hyperparameters}

            if self.model == "LR":
                max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_hyperparameters}

            if self.model == "FLGBM":
                lamb, num_leaves, min_data_in_leaf, max_depth, learning_rate, n_estimators, feature_fraction = [item[1] for item in indiv_list]
                dict_hyperparameters= {'lamb': [lamb], 'num_leaves' : [num_leaves], 'min_data_in_leaf':[min_data_in_leaf], 'max_depth':[max_depth], 'learning_rate': [learning_rate], 'n_estimators': [n_estimators], 'feature_fraction': [feature_fraction]}
                if self.extra != None:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_extra, **dict_test, **dict_extra_test, **dict_hyperparameters}
                else:
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_test, **dict_hyperparameters}
            
            individuals_aux = pd.DataFrame(dict_dataframe)
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            if (first):
                individuals_aux.to_csv('../results/' + self.model + '/' + str(method) + '/individuals/individuals_pareto_' + self.dataset_name + '_seed_' + str(seed) + '_var_' + self.variable_name + '_gen_' + str(self.num_of_generations) + '_indiv_' + str(self.num_of_individuals) + '_model_' + self.model + '_obj_' + self.get_obj_string() + self.get_extra_string() + '.csv', index = False, header = True, columns = list(dict_dataframe.keys()))
            else:
                individuals_aux.to_csv('../results/' + self.model + '/' + str(method) + '/individuals/individuals_pareto_' + self.dataset_name + '_seed_' + str(seed) + '_var_' + self.variable_name + '_gen_' + str(self.num_of_generations) + '_indiv_' + str(self.num_of_individuals) + '_model_' + self.model + '_obj_' + self.get_obj_string() + self.get_extra_string() + '.csv', index = False, mode='a', header=False, columns = list(dict_dataframe.keys()))
    
    #Calculate file with the general pareto front using all pareto fronts in every execution
    def calculate_pareto_optimal(self, seed, runs, method):
        if self.expand:
            pareto_fronts = []
            all_indivs = []
            pareto_optimal = []
            #ATTENTION!!! As we could want to compute the hypervolume, and for returning a structure independent from the measures we use, we should NORMALIZE HERE
            objectives_results_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst'}
            objectives_results_norm_dict = {'num_leaves': 'num_leaves_tst', 'data_weight_avg_depth': 'data_weight_avg_depth_tst'}

            for i in range(runs):
                read = pd.read_csv('../results/' + self.model + '/' + str(method) + '/individuals/individuals_pareto_' + self.dataset_name + '_seed_' + str(seed + i) + '_var_' + self.variable_name + '_gen_' + str(self.num_of_generations) + '_indiv_' + str(self.num_of_individuals) + '_model_' + self.model + '_obj_' + self.get_obj_string() + self.get_extra_string() + '.csv')
                pareto_fronts.append(read)

            hyperparameters = []
            pareto_fronts = pd.concat(pareto_fronts)                            #Union of all pareto fronts got in each run
            pareto_fronts.reset_index(drop=True, inplace=True)                  #Reset index because for each run all rows have repeated ones
            for index, row in pareto_fronts.iterrows():                         #We create an individual object associated with each row
                if self.model == "DT":
                    indiv = IndividualDT()
                    hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
                    indiv.actual_depth = row['actual_depth']
                    indiv.actual_leaves = row['actual_leaves']
                if self.model == "FDT":
                    indiv = IndividualDT()
                    hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'fair_param']
                    indiv.actual_depth = row['actual_depth']
                    indiv.actual_leaves = row['actual_leaves']
                if self.model == "LR":
                    indiv = IndividualLR()
                    hyperparameters = ['max_iter','tol', 'lambda', 'l1_ratio', 'class_weight']
                if self.model == "FLGBM":
                    indiv = IndividualFLGBM()
                    hyperparameters = ['lamb', 'num_leaves', 'min_data_in_leaf', 'max_depth', 'learning_rate', 'n_estimators', 'feature_fraction']
                indiv.features = [row[x] for x in hyperparameters]
                indiv.id = row['id']
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
                    if row['id'] == curr_id:
                        pareto_optimal_df.append(pd.DataFrame({x : row[x] for x in pareto_fronts.columns.tolist()}, index=[0])) #We introduce here the not-normalized version of them
                        found = True
                if not found:
                    pareto_optimal.remove(p)
            #We extract them to a file
            pareto_optimal_df = pd.concat(pareto_optimal_df)
            pareto_optimal_df = pareto_optimal_df.drop_duplicates(subset=(['seed']+hyperparameters), keep='first')
            pareto_optimal_df.to_csv('../results/' + self.model + '/' + str(method) + '/individuals/general_individuals_pareto_' + self.dataset_name + '_baseseed_' + str(seed) + '_nruns_' + str(runs) +'_var_' + self.variable_name + '_gen_' + str(self.num_of_generations) + '_indiv_' + str(self.num_of_individuals) + '_model_' + self.model + '_obj_' + self.get_obj_string() + self.get_extra_string() + '.csv', index = False, header = True, columns = list(pareto_fronts.keys()))

            return pareto_optimal, pareto_optimal_df                   #Population of pareto front individuals