import pandas as pd
from collections import OrderedDict
import random
import numpy as np
import sys

from algorithms.smsemoa.utils import SMSEMOAUtils
from general.population import Population

#Clase que define el funcionamiento del algoritmo NSGA-II
class Evolution:


    def __init__(self, problem, evolutions_df, dataset_name, protected_variable,num_of_generations=5, num_of_individuals=10, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.3, beta_method="uniform"):
        self.utils = SMSEMOAUtils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param, mutation_prob, beta_method)
        self.problem = problem
        self.population = None
        self.evolutions_df = evolutions_df
        self.dataset_name = dataset_name
        self.protected_variable = protected_variable
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method

    #smsemoa METHOD ITSELF
    def evolve(self):
        str_obj = self.problem.objectives[0].__name__
        for i in range(1, len(self.problem.objectives)):
            str_obj += "__" + self.problem.objectives[i].__name__

        random.seed(self.utils.problem.seed)
        np.random.seed(self.utils.problem.seed)
        self.population = self.utils.create_initial_population(self.problem.model) #As the first individual is unrestricted in terms of size, it will have the greatest depth and leave nodes.
        
        ref = []                                                 #Reference for the worst possible result in each objective
        for x in self.utils.problem.objectives:
            if x.__name__ == "num_leaves":                              #Names of the objective functions whose max values can be greater than 1
                ref.append(self.population.population[0].actual_leaves) #Take the max posible value for that population
            else:
                if x.__name__ == "data_weight_avg_depth":                  
                    ref.append(self.population.population[0].actual_depth)  #We add the maximum possible value of that objective for that population
                else:
                    ref.append(1)
        
        self.utils.fast_nondominated_sort(self.population)                  #Calculate rank of all individuals (for tournament purposes)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)                   #Calculate crowding distance for each individual in each front (for tournament purposes)
        child = self.utils.create_child(self.population, self.problem.model)
        for i in range(self.num_of_generations):
            print("GENERATION:",i)
            for k in range(self.num_of_individuals):
                if k == 0:
                    for indiv in self.population.population:
                        dict_general_info = {'id': indiv.id, 'seed': self.problem.seed, 'creation_mode':indiv.creation_mode}
                        dict_objectives= {self.problem.objectives[j].__name__: indiv.objectives[j] for j in range(self.problem.num_of_objectives)}
                        indiv_list = list(indiv.features.items())
                        if self.problem.model == "DT":
                            criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                            dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                            dict_actual_dimensions = {'actual_depth': indiv.actual_depth, 'actual_leaves': indiv.actual_leaves}       #It's really instersting in case of DT to have this size measures
                            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
                        if self.problem.model == "LR":
                            max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                            dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
                            dict_dataframe = {**dict_general_info, **dict_objectives, **dict_hyperparameters}
                        evolutions_aux = pd.DataFrame(dict_dataframe)
                        self.evolutions_df = pd.concat([self.evolutions_df, evolutions_aux])
                if i == (self.num_of_generations-1):
                    self.evolutions_df.to_csv("../results/smsemoa/population/evolution_" + self.dataset_name + '_seed_' + str(self.utils.problem.seed) + "_var_" + self.protected_variable  + "_gen_" + str(self.num_of_generations) + "_indiv_" + str(self.num_of_individuals) + '_model_' + self.problem.model + '_obj_' + str_obj + ".csv", index = False, header = True, columns = list(dict_dataframe.keys()))
                self.population.append(child)
                self.population = self.utils.reduce(self.population, ref)         #Erase individual in the worst front whose hypervolume contribution is minimal
                self.utils.fast_nondominated_sort(self.population)                  #Calculate rank of all individuals (for tournament purposes)
                for front in self.population.fronts:
                    self.utils.calculate_crowding_distance(front)                   #Calculate crowding distance for each individual in each ffront (for tournament purposes)
                child = self.utils.create_child(self.population, self.problem.model)
                
                dict_general_info = {'id': child.id, 'seed': self.problem.seed, 'creation_mode':child.creation_mode}
                dict_objectives= {self.problem.objectives[j].__name__: child.objectives[j] for j in range(self.problem.num_of_objectives)}
                indiv_list = list(child.features.items())
                if self.problem.model == "DT":
                    criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                    dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                    dict_actual_dimensions = {'actual_depth': child.actual_depth, 'actual_leaves': child.actual_leaves}       #It's really instersting in case of DT to have this size measures
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
                if self.problem.model == "LR":
                    max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                    dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_hyperparameters}
                evolutions_aux = pd.DataFrame(dict_dataframe)
                self.evolutions_df = pd.concat([self.evolutions_df, evolutions_aux])
        return self.population.fronts[0]
