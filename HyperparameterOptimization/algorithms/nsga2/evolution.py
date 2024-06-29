import pandas as pd
import random
import numpy as np
import os
import time
import copy

from algorithms.nsga2.utils import NSGA2Utils
from general.population import Population
from general.ml import create_generation_stats, save_generation_stats
from joblib import Parallel, delayed


PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + '/results/'
class Evolution:
    """
    Class defining NSGA-II multiobjective evolutionary algorithm
    """

    def __init__(self, problem, evolutions_df, dataset_name, model_name, protected_variable,num_of_generations=5 ,num_of_individuals=10, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.3, beta_method="uniform"):
        """
        Constructor
            Paramaeters:
                - problem: Problem class which defines teh problem
                - evolutions_df: Empty Dataframe to store evolution data
                - dataset_name: Name of the dataset
                - model_name: Name of the learning algorithm
                - protected_variable: Name of the protected variable
                - num_of_generations: Number of generations
                - num_of_individuals: Number of individuals
                - num_of_tour_participants: Number of participants in tournament
                - tournament_prob: Probability value for tournament to select the best individual
                - crossover_param: Parameter which controls the crossover method
                - mutation_param: Parameter which controls the mutation method
                - mutation_prob: Probability of mutations to happen
                - beta_method: Beta method calculation for crossover (possible values are 'uniform' or anything other than 'uniform')
        """
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param, mutation_prob, beta_method)
        self.problem = problem
        self.population = None
        self.evolutions_df = evolutions_df
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.protected_variable = protected_variable
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method

    def get_dataframes(self, indiv):
        """
        Get dataframes for saving data about individuals
            Parameters:
                -indiv: individual for which to generate the datarame
            Returns:
                - Dataframe containing the information of the inndividual
        """
        dict_general_info = {'ID': indiv.id, 'seed': self.problem.seed, 'creation_mode':indiv.creation_mode}
        dict_objectives= {f"{self.problem.objectives[j].__name__}_val": indiv.objectives[j] for j in range(self.problem.num_of_objectives)}
        indiv_list = list(indiv.features.items())
        if self.problem.model == "DT":
            criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
            dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
            dict_actual_dimensions = {'leaves': indiv.actual_leaves, 'depth': indiv.actual_depth, 'data_avg_depth': indiv.actual_data_avg_depth, 'depth_unbalance': indiv.actual_depth_unbalance}       #It's really instersting in case of DT to have this size measures
            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
        if self.problem.model == "FDT":
            criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight, fair_param = [item[1] for item in indiv_list]
            dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'fair_param': [fair_param]}
            dict_actual_dimensions = {'leaves': indiv.actual_leaves, 'depth': indiv.actual_depth, 'data_avg_depth': indiv.actual_data_avg_depth, 'depth_unbalance': indiv.actual_depth_unbalance}       #It's really instersting in case of DT to have this size measures
            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
        if self.problem.model == "LR":
            max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
            dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_hyperparameters}
        if self.problem.model == "FLGBM":
            num_leaves, min_data_in_leaf, max_depth, learning_rate, n_estimators, feature_fraction, class_weight, fair_param= [item[1] for item in indiv_list]
            dict_hyperparameters= {'num_leaves' : [num_leaves], 'min_data_in_leaf':[min_data_in_leaf], 'max_depth':[max_depth], 'learning_rate': [learning_rate], 'n_estimators': [n_estimators], 'feature_fraction': [feature_fraction], 'class_weight': [class_weight], 'fair_param': [fair_param]}
            dict_actual_dimensions = {'n_estimators': indiv.actual_n_estimators, 'n_features': indiv.actual_n_features, 'feature_importance_std': indiv.actual_feature_importance_std}
            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
        return pd.DataFrame(dict_dataframe)
    
    def evolve(self):
        """
        Applies NSGA-II method itself
            Returns:
                - return_pop: List containing Pareto-optimal individuals from the last population
        """
        str_obj = self.problem.objectives[0].__name__
        for i in range(1, len(self.problem.objectives)):
            str_obj += "__" + self.problem.objectives[i].__name__

        random.seed(self.utils.problem.seed)
        np.random.seed(self.utils.problem.seed)
        self.population = self.utils.create_initial_population(self.problem.model)
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population, self.problem.model)

        start_process_time = time.process_time()
        start_total_time = time.time()
        generations_df = create_generation_stats(self.model_name)

        for i in range(self.num_of_generations):

            gen_df = self.evolutions_df.copy(deep=True).iloc[0:0]
            results_df = pd.concat(Parallel(n_jobs=-1)(delayed(self.get_dataframes)(indiv) for indiv in self.population.population))                
            gen_df = pd.concat([gen_df, results_df])
            self.evolutions_df = pd.concat([self.evolutions_df, gen_df])
            #print(self.evolutions_df)
            if i == (self.num_of_generations-1):
                self.evolutions_df.to_csv(f"{PATH_TO_RESULTS}{self.model_name}/nsga2/population/{self.dataset_name}/{self.dataset_name}_seed_{self.utils.problem.seed}_var_{self.protected_variable}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.problem.model}_obj_{str_obj}{self.problem.get_extra_string()}.csv", index = False, header = True, columns = list(self.evolutions_df.keys()))
            generations_df = save_generation_stats(generations_df, gen_df, self.problem.model, time.process_time() - start_process_time, time.time() - start_total_time)
            start_process_time = time.process_time()
            start_total_time = time.time()
            print("GENERATION:",i+1)
            self.population.extend(children)


            # Remove individuals with repeated objective functions:
            new_pop = []
            new_pop_initial_size = 0

            all_objectives = []
            for indiv in self.population:
                if not indiv.objectives in all_objectives:
                    all_objectives.append(indiv.objectives)
                    new_pop.append(indiv)
                    new_pop_initial_size += 1
            
            while len(new_pop) < self.num_of_individuals:
                for i in range(new_pop_initial_size):
                    new_pop.append(copy.deepcopy(new_pop[i]))

            self.population.population = new_pop
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) < self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            self.population = new_population
            if i < (self.num_of_generations - 1):
                children = self.utils.create_children(self.population, self.problem.model)
        
        self.utils.fast_nondominated_sort(self.population)
        generations_df.to_csv(f"{PATH_TO_RESULTS}{self.model_name}/nsga2/generation_stats/{self.dataset_name}/{self.dataset_name}_seed_{self.utils.problem.seed}_var_{self.protected_variable}_gen_{self.num_of_generations}_indiv_{self.num_of_individuals}_model_{self.problem.model}_obj_{str_obj}{self.problem.get_extra_string()}.csv", index = False, header = True)
        self.utils.fast_nondominated_sort(self.population)  #Once we've finished, let's return only nondominated individuals
        #Calculate test objectives
        return_pop = Parallel(n_jobs=-1)(delayed(self.utils.parallel_calc_objectives)(child, first=False, calc_test=True) for child in self.population.fronts[0])
        for indiv in return_pop:
            print(indiv.objectives_test)
        print(len(return_pop))
        
        return return_pop