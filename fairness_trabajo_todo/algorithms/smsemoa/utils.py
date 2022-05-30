from general.population import Population
from general.individual import *
import random
from general.ml import *
import pygmo as pg

#Funciones de utilidades para el uso del algoritmo NSGA-II
#Aquí habrá funciones básicas de NSGA-II que simplifiquen el código del método evolve de la clase Evolution
class SMSEMOAUtils:

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

    #Creación de la población inicial. 
    def create_initial_population(self, model):
        population = Population()
        first_individual = True
        for k in range(self.num_of_individuals):        #There will be at least 1 Gini and 1 Entropy individuals in this initial population
            if model == "DT":
                if k == 0:
                    individual = self.problem.generate_default_individual_gini()
                elif k == 1:
                    individual = self.problem.generate_default_individual_entropy()
                else:
                    individual = self.problem.generate_individual()
            
            if model == "LR":
                if k == 0:
                    individual = self.problem.generate_first_default_lr()
                elif k == 1:
                    individual = self.problem.generate_second_default_lr()
                else:
                    individual = self.problem.generate_individual()
            
            self.problem.calculate_objectives(individual, first_individual, self.problem.seed, 'smsemoa')
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


    #Cálculo de la distancia de crowding, DENTRO DE UN MISMO FRENTE.
    def calculate_crowding_distance(self, front):
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
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    #Generates only 1 child for smsemoa, using SBX crossover and polynomial mutation
    def create_child(self, population, model):
        first_individual = False
        parent1 = self.__tournament(population)                                 #Selection
        parent2 = parent1
        while parent1 == parent2:
            parent2 = self.__tournament(population)
        child = []
        eta_parameter = 10; prob_mut = 0.5; prob_mut_gene = 1./self.problem.num_of_variables
        child = self.sbx_cross(parent1, parent2, eta_parameter, self.problem)
        if random.uniform(0,1) < 0.5:
            child = child[0]
        else:
            child = child[1]
        child.creation_mode = "crossover"
        if random.uniform(0,1) < prob_mut:
            child = self.polynomial_mutation(child, eta_parameter, prob_mut_gene, self.problem)
            child.creation_mode = "mutation"
        child.features = decode(self.problem.variables_range, model, **child.features)             #Adjust features to their range
        self.problem.calculate_objectives(child, first_individual, self.problem.seed, 'smsemoa')
        return child

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

    def get_maxcomplexity(self,prob):
        return prob.maxcomplexity
    
    #Return 2 individuals using sbx crossover
    #The method is made for continuous variables, for discrete ones, disctretization is needed
    #Adaptation from code at https://github.com/DEAP/deap
    def sbx_cross(self, parent1, parent2, eta, prob):
        ind1 = self.problem.generate_individual()
        ind2 = self.problem.generate_individual()
        up = []
        low = []
        for i in range(0, prob.num_of_variables):
            low.append(prob.variables_range[i][0])
            up.append(prob.variables_range[i][1])
        for hyperparameter, xl, xu in zip(parent1.features, low, up):
            if parent1.features[hyperparameter] == None and parent2.features[hyperparameter] == None:      #Some attributes can have "None" value
                ind1.features[hyperparameter] = None
                ind2.features[hyperparameter] = None
            elif parent1.features[hyperparameter] == None or parent2.features[hyperparameter] == None:
                u = random.random()
                if u <= 0.5:
                    ind1.features[hyperparameter] = parent1.features[hyperparameter]
                    ind2.features[hyperparameter] = parent2.features[hyperparameter]
                else:
                    ind1.features[hyperparameter] = parent2.features[hyperparameter]
                    ind2.features[hyperparameter] = parent1.features[hyperparameter]
            else:
                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                if abs(ind1.features[hyperparameter] - ind2.features[hyperparameter]) > 1e-14:
                    x1 = min(ind1.features[hyperparameter], ind2.features[hyperparameter])
                    x2 = max(ind1.features[hyperparameter], ind2.features[hyperparameter])
                    rand = random.random()

                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                    c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)

                    if random.random() <= 0.5:
                        ind1.features[hyperparameter] = c2
                        ind2.features[hyperparameter] = c1
                    else:
                        ind1.features[hyperparameter] = c1
                        ind2.features[hyperparameter] = c2
        return ind1, ind2
    

    #indp is mutation probability for each variable
    #eta is a factor that controls convergence or divergence to the base solution
    #Adaptation from https://github.com/DEAP/deap in tools
    def polynomial_mutation(self, individual, eta, indpb, prob):
        low = []
        up = []
        for i in range(0, prob.num_of_variables):
            low.append(prob.variables_range[i][0])
            up.append(prob.variables_range[i][1])
        for hyperparameter, xl, xu in zip(individual.features, low, up):
            if not(individual.features[hyperparameter] == None) and random.random() <= indpb:
                x = individual.features[hyperparameter]
                if xu-xl == 0:
                    delta_1 = 0
                    delta_2 = 0
                else:
                    delta_1 = (x - xl) / (xu - xl)
                    delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                individual.features[hyperparameter] = x
        return individual

    #Gets rid of the individual belonging to the worst dominance set whose hypervolume contribution is minimal
    def reduce(self, population, ref):
        #We're going to use pygmo for it. A previous implementation used the method hypervolume_increment
        hv = pg.hypervolume([x.objectives for x in population.population])  #Objectives of each element in the population
        rem_index = hv.least_contributor(ref)                               #Index of the individual with the least contribution to the hypervolume
        population.population.pop(rem_index)                                #Remove it
        return population