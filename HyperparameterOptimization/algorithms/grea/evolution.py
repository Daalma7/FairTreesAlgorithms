import pandas as pd
from collections import OrderedDict
import random
import numpy as np

from algorithms.grea.utils import greaUtils
from general.population import PopulationGrea
from general.individual import IndividualDTGrea, IndividualLRGrea

#Clase que define el funcionamiento del algoritmo NSGA-II
class Evolution:


    def __init__(self, problem, evolutions_df, dataset_name, protected_variable,num_of_generations=5, num_of_individuals=10, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.3, beta_method="uniform", div = 10):
        self.problem = problem
        self.utils = greaUtils(problem, num_of_individuals, tournament_prob, crossover_param, mutation_param, mutation_prob, beta_method)
        self.population = None
        self.evolutions_df = evolutions_df
        self.dataset_name = dataset_name
        self.protected_variable = protected_variable
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method
        self.div = div

    #grea METHOD ITSELF
    def evolve(self):
        str_obj = self.problem.objectives[0].__name__
        for i in range(1, len(self.problem.objectives)):
            str_obj += "__" + self.problem.objectives[i].__name__

        random.seed(self.utils.problem.seed)
        np.random.seed(self.utils.problem.seed)
        self.population = self.utils.create_initial_population(self.div, self.problem.model)
        for i in range(self.num_of_generations):
            for indiv in self.population.population:
                dict_general_info = {'id': indiv.id, 'seed': self.problem.seed, 'creation_mode':indiv.creation_mode}
                dict_objectives= {self.problem.objectives[j].__name__: indiv.objectives[j] for j in range(self.problem.num_of_objectives)}
                indiv_list = list(indiv.features.items())
                if self.problem.model == "DT":
                    criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
                    dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight]}
                    dict_actual_dimensions = {'actual_depth': indiv.actual_depth, 'actual_leaves': indiv.actual_leaves}       #It's really instersting in case of DT to have this size measures
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
                if self.problem.model == "FDT":
                    criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight, fair_param = [item[1] for item in indiv_list]
                    dict_hyperparameters = {'criterion': [criterion], 'max_depht': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'fair_param': [fair_param]}
                    dict_actual_dimensions = {'actual_depth': indiv.actual_depth, 'actual_leaves': indiv.actual_leaves}       #It's really instersting in case of DT to have this size measures
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_actual_dimensions, **dict_hyperparameters}
                if self.problem.model == "LR":
                    max_iter, tol, lambd, l1_ratio, class_weight = [item[1] for item in indiv_list]
                    dict_hyperparameters= {'max_iter': [max_iter], 'tol': [tol], 'lambda': [lambd], 'l1_ratio': [l1_ratio], 'class_weight': [class_weight]}
                    dict_dataframe = {**dict_general_info, **dict_objectives, **dict_hyperparameters}
                evolutions_aux = pd.DataFrame(dict_dataframe)
                self.evolutions_df = pd.concat([self.evolutions_df, evolutions_aux])
            if i == (self.num_of_generations-1):
                self.evolutions_df.to_csv("../results/grea/population/evolution_" + self.dataset_name + '_seed_' + str(self.utils.problem.seed) + "_var_" + self.protected_variable  + "_gen_" + str(self.num_of_generations) + "_indiv_" + str(self.num_of_individuals) + '_model_' + self.problem.model + '_obj_' + str_obj + ".csv", index = False, header = True, columns = list(dict_dataframe.keys()))
            print("GENERATION:",i)
            self.population.calculate_grid_boundries()  #Calculate grid environment and locations of its individuals
            self.population.calculate_grid_locations()  #Grid coordinate calculation is done here
            for indiv in self.population.population:                    #Calculate quality measures for each individual
                indiv.calculate_grid_crowding_distance(self.population)             #Only this measure, appart from grid locations, will be relevant for offspring generation
            children = self.utils.create_children(self.population, self.problem.model) #Create children population of equal size of current population and make mutations
            self.population.extend(children)
            self.population = self.utils.environmental_selection(self.population) #Environmental selection
        
        self.utils.fast_nondominated_sort(self.population)
        return self.population.fronts[0]
        