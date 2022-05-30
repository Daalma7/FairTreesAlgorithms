from general.population import PopulationGrea
import random
from general.ml import *

#Utility functions for GrEA algorithm
class greaUtils:

    def __init__(self, problem, num_of_individuals=50,
                tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.1, beta_method='uniform'):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method

    #Initial population generation
    def create_initial_population(self, div, model):
        population = PopulationGrea(div)
        first_individual = True
        for k in range(self.num_of_individuals):        #There will be at least 1 Gini and 1 Entropy individuals in this initial population
            if model == "DT":
                if k == 0:
                    individual = self.problem.generate_default_individual_gini('grea')
                elif k == 1:
                    individual = self.problem.generate_default_individual_entropy('grea')
                else:
                    individual = self.problem.generate_individual('grea')
            
            if model == "LR":
                if k == 0:
                    individual = self.problem.generate_first_default_lr('grea')
                elif k == 1:
                    individual = self.problem.generate_second_default_lr('grea')
                else:
                    individual = self.problem.generate_individual('grea')
            
            self.problem.calculate_objectives(individual, first_individual, self.problem.seed, 'grea')
            first_individual = False
            population.append(individual)          
        return population

    #Ordenación de población por dominancia entre soluciones.
    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:               #Establecimiento mejor frente e info de dominación para generar los demás
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):                  #Si la solución actual domina a la otra
                    individual.dominated_solutions.append(other_individual) #Se añade a su lista de soluciones dominadas
                elif other_individual.dominates(individual):                #Si no
                    individual.domination_count += 1                        #Aumentamos el contador de dominación
            if individual.domination_count == 0:                            #Si no es dominada por nadie
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:        #Mientras que haya soluciones en el frente actual considerado
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:     #Para cada individuo dominado por la solución actual
                    other_individual.domination_count -= 1                  #Le quitamos 1 al contador
                    if other_individual.domination_count == 0:              #Si llega a 0 no hay sols que las dominen y entra en el siguiente frente
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)
        return population


    #Generates the offspring of the current population
    def create_children(self, population, model):
        first_individual = False
        children = []
        while len(children) < len(population):  #We want to have the same amount of children as of parents
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2, model)
            prob_mutation_child1 = random.uniform(0,1)
            prob_mutation_child2 = random.uniform(0,1)
            if prob_mutation_child1 < self.mutation_prob:
                self.__mutate(child1, self.mutation_prob, model)
                child1.creation_mode = "mutation"
            if prob_mutation_child2 < self.mutation_prob:
                self.__mutate(child2, self.mutation_prob, model)
                child2.creation_mode = "mutation"
            child1.features = decode(self.problem.variables_range, model, **child1.features)
            child2.features = decode(self.problem.variables_range, model, **child2.features)
            self.problem.calculate_objectives(child1, first_individual, self.problem.seed, 'grea')
            self.problem.calculate_objectives(child2, first_individual, self.problem.seed, 'grea')
            children.append(child1)
            children.append(child2)
        return children

    #Basic crossover function. The same as in NSGA-II
    def __crossover(self, individual1, individual2, model):
        child1 = self.problem.generate_individual('grea')
        child2 = self.problem.generate_individual('grea')
        child1.creation_mode = "crossover"
        child2.creation_mode = "crossover"
        for hyperparameter in child1.features:
            hyperparameter_index = list(child1.features.keys()).index(hyperparameter)
            if self.beta_method == 'uniform':
                beta = self.__get_beta_uniform()
            else:
                beta = self.__get_beta()
            if individual1.features[hyperparameter] is None and individual2.features[hyperparameter] is None:
               child1.features[hyperparameter] = individual1.features[hyperparameter]
               child2.features[hyperparameter] = individual2.features[hyperparameter]
            elif individual1.features[hyperparameter] is None or individual2.features[hyperparameter] is None:
                u = random.random()
                if u <= 0.5:
                    child1.features[hyperparameter] = individual1.features[hyperparameter]
                    child2.features[hyperparameter] = individual2.features[hyperparameter]
                else:
                    child1.features[hyperparameter] = individual2.features[hyperparameter]
                    child2.features[hyperparameter] = individual1.features[hyperparameter]
            else:
                x1 = (individual1.features[hyperparameter] + individual2.features[hyperparameter])/2
                x2 = abs((individual1.features[hyperparameter] - individual2.features[hyperparameter])/2)
                child1.features[hyperparameter] = x1 + beta*x2
                child2.features[hyperparameter] = x1 - beta*x2
                if child1.features[hyperparameter] < self.problem.variables_range[hyperparameter_index][0]:
                   child1.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][0]
                elif child1.features[hyperparameter] > self.problem.variables_range[hyperparameter_index][1]:
                   child1.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][1]
                if child2.features[hyperparameter] < self.problem.variables_range[hyperparameter_index][0]:
                   child2.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][0]
                elif child2.features[hyperparameter] > self.problem.variables_range[hyperparameter_index][1]:
                   child2.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][1]
        return child1, child2

    def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2*u)**(1/(self.crossover_param+1))
        return (2*(1-u))**(-1/(self.crossover_param+1))

    def __get_beta_uniform(self):
        u = random.uniform(0, 0.5)
        return u

    # The same as in NSGA-II
    def __mutate(self, child, prob_mutation, model):
        hyperparameter = random.choice(list(child.features))
        hyperparameter_index = list(child.features.keys()).index(hyperparameter)
        u, delta = self.__get_delta()
        if child.features[hyperparameter] is not None:
            if u < 0.5:
                child.features[hyperparameter] += delta*(child.features[hyperparameter] - self.problem.variables_range[hyperparameter_index][0])
                child.features = decode(self.problem.variables_range, model, **child.features)
            else:
                child.features[hyperparameter] += delta*(self.problem.variables_range[hyperparameter_index][1] - child.features[hyperparameter])
                child.features = decode(self.problem.variables_range, model, **child.features)
            if child.features[hyperparameter] < self.problem.variables_range[hyperparameter_index][0]:
                child.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][0]
            elif child.features[hyperparameter] > self.problem.variables_range[hyperparameter_index][1]:
                child.features[hyperparameter] = self.problem.variables_range[hyperparameter_index][1]


    def __get_delta(self):
        u = random.random()
        if u <= 0.5:
            return u, (2*u)**(1/(self.mutation_param + 1)) - 1
        return u, 1 - (2*(1-u))**(1/(self.mutation_param + 1))


    def __tournament(self, population):
        p = random.sample(population.population, 2)
        #First tournament criterion: standard or grid domination
        if p[0].dominates(p[1]) or p[0].grid_dominates(p[1]):
            return p[0]
        if p[1].dominates(p[0]) or p[1].grid_dominates(p[0]):
            return p[1]
        #Second tournament criterion: grid crowding distance
        if p[0].grid_crowding_distance < p[1].grid_crowding_distance:
            return p[0]
        if p[1].grid_crowding_distance < p[0].grid_crowding_distance:
            return p[1]
        #Third tournament criterion: randomly seleced with equal probability
        if self.__choose_with_prob(0.5):
            return p[0]
        else:
            return p[1]

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

    def get_maxcomplexity(self, prob):
        return prob.maxcomplexity



    #Does the environmental selection for the current population passed in pop
    def environmental_selection(self, pop):
        pop = self.fast_nondominated_sort(pop)                #Create non-dominated fronts
        new_population = PopulationGrea(pop.div)
        front_num = 0
        while len(new_population.population) + len(pop.fronts[front_num]) <= self.num_of_individuals: #Pick the best fronts until the next we had to take would fill up the new population
            new_population.extend(pop.fronts[front_num])
            front_num += 1
        if len(new_population.population) == len(pop.population):
            return new_population
        front_population = PopulationGrea(pop.div)
        front_population.extend(pop.fronts[front_num])
        front_population.calculate_grid_boundries()     #Initializes grid environment BUT grid crowding distance is set to 0 as is updated as elements are pick
        front_population.calculate_grid_locations()
        for indiv in front_population.population:
            indiv.calculate_grid_rating()
            indiv.calculate_grid_coordinate_point_distance(front_population)
            indiv.grid_crowding_distance = 0
        while len(new_population.population) < self.num_of_individuals:
            best = front_population.findout_best()
            new_population.append(best)                 #Add it to the new population
            front_population.population.remove(best)    #Remove it from the front one
            best.calculate_grid_crowding_distance(front_population)
            front_population.GR_adjustment(best)
        
        return new_population