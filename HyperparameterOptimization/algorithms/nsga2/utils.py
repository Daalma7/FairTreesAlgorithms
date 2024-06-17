from general.population import Population
import random
from general.ml import decode
from joblib import Parallel, delayed


class NSGA2Utils:
    """
    Class definint NSGA2 utilities
    """

    def __init__(self, problem, num_of_individuals=50,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.1, beta_method='uniform'):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method

    def parallel_create_initial_population_single(self):
        """
        Auxiliary function for creating initial population for parallel executions
        """
        individual = self.problem.generate_individual()
        self.problem.calculate_objectives(individual, False, False)
        return individual

    def create_initial_population(self, model, parallel=True):
        """
        Creates initial population
            Parameters:
                - model: Algorithm used
                - parallel: Specifies if individuals will be created using parallelism or not
            Returns:
                - population: Created new population
        """
        if parallel:
            population = Population()
            indivlist = []
            if model == 'DT':
                individual = self.problem.generate_default_individual_dt(criterion='gini')
                self.problem.calculate_objectives(individual, True, False)
                population.append(individual)
                individual = self.problem.generate_default_individual_dt(criterion='entropy')
                self.problem.calculate_objectives(individual, False, False)
                population.append(individual)
            elif model == 'FDT':
                individual = self.problem.generate_default_individual_fdt(criterion='gini')
                self.problem.calculate_objectives(individual, True, False)
                population.append(individual)
                individual = self.problem.generate_default_individual_fdt(criterion='entropy')
                self.problem.calculate_objectives(individual, False, False)
                population.append(individual)
            elif model == 'LR':
                individual = self.problem.generate_default_individual_lr(num='first')
                self.problem.calculate_objectives(individual, True, False)
                population.append(individual)
                individual = self.problem.generate_default_individual_lr(num='second')
                self.problem.calculate_objectives(individual, False, False)
                population.append(individual)
            elif model == 'FLGBM':
                individual = self.problem.generate_default_individual_flgbm(num='first')
                self.problem.calculate_objectives(individual, True, False)
                population.append(individual)
                individual = self.problem.generate_default_individual_flgbm(num='second')
                self.problem.calculate_objectives(individual, False, False)
                population.append(individual)
            new_pop = Parallel(n_jobs=-1)(delayed(self.parallel_create_initial_population_single)() for i in range(self.num_of_individuals - 2))
            for indiv in new_pop:
                population.append(indiv)
        else:
            population = Population()
            first_individual = True
            for k in range(self.num_of_individuals):        #There will be at least 1 Gini and 1 Entropy individuals in this initial population
                if model == "DT":
                    if k == 0:
                        individual = self.problem.generate_default_individual_dt(criterion='gini')
                    elif k == 1:
                        individual = self.problem.generate_default_individual_dt(criterion='entropy')
                    else:
                        individual = self.problem.generate_individual()
                
                elif model == "FDT":
                    if k == 0:
                        individual = self.problem.generate_default_individual_fdt(criterion='gini')
                    elif k == 1:
                        individual = self.problem.generate_default_individual_fdt(criterion='entropy')
                    else:
                        individual = self.problem.generate_individual()
                
                elif model == "LR":
                    if k == 0:
                        individual = self.problem.generate_default_lr(num='first')
                    elif k == 1:
                        individual = self.problem.generate_default_lr(num='second')
                    else:
                        individual = self.problem.generate_individual()
                
                elif model == "FLGBM":
                    if k == 0:
                        individual = self.problem.generate_default_flgbm(num='first')
                    elif k == 1:
                        individual = self.problem.generate_default_flgbm(num='second')
                    else:
                        individual = self.problem.generate_individual()
                self.problem.calculate_objectives(individual, first_individual, False)
                first_individual = False
                population.append(individual)

        return population
    
    def fast_nondominated_sort(self, population):
        """
        Sorting population criterion by dominance of solutions. Population is sorted dividing it in different
        fronts, where all individuals from front i dominates all solutions from the rest of the fronts
            Parameters:
                - Population: population to sort. Fronts are asigned to a population attribute
        """
        population.fronts = [[]]

        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):                  # If the current solution dominates the other
                    individual.dominated_solutions.append(other_individual) # It is appended to its list of dominated solutions
                elif other_individual.dominates(individual):                # If the other solution dominates the current one
                    individual.domination_count += 1                        # We increase its domination count

        for individual in population:
            if individual.domination_count == 0:                            # If it is not dominated by any solution
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0

        while len(population.fronts[i]) > 0:        # While there are still solutions in the current front
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:     # For each individual dominated by the current solution
                    other_individual.domination_count -= 1                  # The domination count decreases by 1
                    if other_individual.domination_count == 0:              # If it reaches 0 there are no solutions which dominates them and it will fall in the current front
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)

    #Calculation of crowding_distance given a front
    def calculate_crowding_distance(self, front):
        """
        Calculates crowding distance given a front. This distance assigns each individual a value depending on how it
        is surrounded by other solutions, being better if there are no other solutions near it.
            Parameters:
                - front: Given front, for which calculate crowding distance to all its individuals
        """
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10**9
                front[solutions_num-1].crowding_distance = 10**9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num-1):
                    front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m])/scale

    def crowding_operator(self, individual, other_individual):
        """
        Auxiliary operator for tournament, comparing crowding distance of 2 individuals:
            Parameters:
                - individual: First of the individuals to compare
                - other_individual: Second of the individuals to compare
            Return:
                - 1 if the first individual is preferred, -1 if not
        """
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def parallel_create_children(self, population, model):
        """
        Auxiliary function to create children population using a parallel execution
            Parameters:
                - Population: Population containing previous population
                - model: ML model type
            Returns:
                - [child1, child2]: Two children created
        """
        parent1 = self.__tournament(population)
        parent2 = parent1
        while parent1 == parent2:
            parent2 = self.__tournament(population)
        child1, child2 = self.__crossover(parent1, parent2, model)
        prob_mutation_child1 = random.uniform(0,1)
        prob_mutation_child2 = random.uniform(0,1)
        if(prob_mutation_child1 < self.mutation_prob):
            self.__mutate(child1, model)
            child1.creation_mode = "mutation"
        if(prob_mutation_child2 < self.mutation_prob):
            self.__mutate(child2, model)
            child2.creation_mode = "mutation"
        child1.features = decode(self.problem.variables_range, model, **child1.features)
        child2.features = decode(self.problem.variables_range, model, **child2.features)
        return [child1, child2]
    
    def parallel_calc_objectives(self, child, first=False, calc_test=False):
        """
        Auxiliary function to calculate objectives of a given individual
            Parameters:
                - child: individual to calculate objectives
                - first: Boolean value indicating if the individual is the first individual or not
            Returns:
                - child: Individual with calculated objectives
        """
        self.problem.calculate_objectives(child, first, calc_test)
        return child
    
    def create_children(self, population, model, parallel = True):
        """
        Creates a children population
            Parameters:
                - population: Population of parent individuals
                - model: ML model which each individual represents
                - parallel: Boolean value specifying if the children generation will be done in parallel or sequentially
            Returns:
                - Children population
        """
        if parallel:
            pop_size = len(population.population)
            child_pop = Parallel(n_jobs=-1)(delayed(self.parallel_create_children)(population, model) for i in range(pop_size))
            child_pop = [child for children in child_pop for child in children]
            child_pop_prev = [child for child in child_pop if child.calc_objectives]
            child_pop_new = Parallel(n_jobs=-1)(delayed(self.parallel_calc_objectives)(child) for child in child_pop if not child.calc_objectives)
            return child_pop_prev + child_pop_new
        else:
            ret = []
            for i in range(int(len(population.population)/2)):
                parent1 = self.__tournament(population)
                parent2 = parent1
                while parent1 == parent2:
                    parent2 = self.__tournament(population)
                child1, child2 = self.__crossover(parent1, parent2, model)
                prob_mutation_child1 = random.uniform(0,1)
                prob_mutation_child2 = random.uniform(0,1)
                if(prob_mutation_child1 < self.mutation_prob):
                    self.__mutate(child1, model)
                    child1.creation_mode = "mutation"
                if(prob_mutation_child2 < self.mutation_prob):
                    self.__mutate(child2, model)
                    child2.creation_mode = "mutation"
                child1.features = decode(self.problem.variables_range, model, **child1.features)
                child2.features = decode(self.problem.variables_range, model, **child2.features)
                ret.append(child1)
                ret.append(child2)
            return ret

    def __crossover(self, individual1, individual2, model):
        """
        Crossover function between 2 individuals. It uses the NSGA2 paradigm
            Parameters:
                - individual1: First individual to apply crossover
                - individual2: Second individual to apply crossover
                - model: ML model which each individual represents
            Return:
                - child1, child2: Individuals with crossover already applied
        """
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
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
        child1.features = decode(self.problem.variables_range, model, **child1.features)
        child2.features = decode(self.problem.variables_range, model, **child2.features)
        return child1, child2

    def __get_beta(self):
        """
        Auxiliary function to get beta value for crossover
            Returns:
                - Beta value
        """
        u = random.random()
        if u <= 0.5:
            return (2*u)**(1/(self.crossover_param+1))
        return (2*(1-u))**(-1/(self.crossover_param+1))

    def __get_beta_uniform(self):
        """
        Auxiliary function to get an uniform value for the beta parameter
            Returns:
                - Beta value
        """
        u = random.uniform(0, 1)
        return u

    def __mutate(self, child, model):
        """
        Mutation criterion, following NSGA2 procedures
            Parameters:
                - child: Individual to mutate
                - model: ML model which that individual represents
        """
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
        """
        Auxiliary function to get beta value for mutation
            Returns:
                - Beta value
        """
        u = random.random()
        if u <= 0.5:
            return u, (2*u)**(1/(self.mutation_param + 1)) - 1
        return u, 1 - 2*((1-u)**(1/(self.mutation_param + 1)))

    def __tournament(self, population):
        """
        Tournament criterion for the population, to select parents population
            Parameters:
                - population: Initial population to select parents
            Returns:
                - Parents population
        """
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        """
        Auxiliary function, which selects with probability prob
            Parameters:
                - prob: The probability of choosing (should be in range (0,1))
            Returns:
                - Boolean value, with True meaning select, and False meaning do not select
        """
        if random.random() <= prob:
            return True
        return False

    def get_maxcomplexity(self,prob):
        """
        Auxiliary function which is not currently used
        """
        return prob.maxcomplexity