import math
import abc

############################################################################
# Base individual, don't represent any particular ML classification method #
############################################################################

class Individual(object):

    def __init__(self):
        self.id = None
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None
        self.extra = None
        self.creation_mode = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates_standard(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)


    @abc.abstractmethod
    def dominates(self, other_individual):
        pass


################################################################
# This class represents an individual which is a Decision Tree #
################################################################

class IndividualDT(Individual):

    def __init__(self):
        super().__init__()
        self.actual_depth = None
        self.actual_leaves = None

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        eq_condition = True
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
            eq_condition = eq_condition and first == second
            
        if (eq_condition):
            if ((self.features['max_leaf_nodes'] is None) or (other_individual.features['max_leaf_nodes'] is None)):
                return (self.actual_leaves < other_individual.actual_leaves)
            else:
                return ((self.actual_leaves < other_individual.actual_leaves) or
                       ((self.actual_leaves == other_individual.actual_leaves) and (self.features['max_leaf_nodes'] < other_individual.features['max_leaf_nodes'])))
        else:
            return (and_condition and or_condition)


###################################################
# Individual Decision Tree for the GrEA algorithm #
###################################################

class IndividualDTGrea(IndividualDT):

    def __init__(self):
        super().__init__()
        self.grid_coordinates = None
        self.grid_rating = None
        self.grid_crowding_distance = None
        self.grid_coordinate_point_distance = None
        self.punishment_degree = 0

    #For grid dominance
    def grid_dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.grid_coordinates, other_individual.grid_coordinates):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

    #Returns grid difference between individuals, having calculated their locations
    def grid_difference(self, other_individual):
        sum = 0
        for k in range(0, len(self.objectives)):
            sum += abs(self.grid_coordinates[k]-other_individual.grid_coordinates[k])
        return sum

    #Grid rating measure for convergence
    #This is a convergence measure
    def calculate_grid_rating(self):
        sum = 0
        for k in range(0, len(self.grid_coordinates)):
            sum += self.grid_coordinates[k]
        self.grid_rating = sum


    #Calculates grid crowding distance (GCD) concerning all the population
    #if the individual belongs to the population itself, at least GCD = M
    #This is a diversity measure
    def calculate_grid_crowding_distance(self, population):
        M = len(self.objectives)    #Constant which depend on the number of objective functions
        sum = 0
        for indiv in population:
            g = self.grid_difference(indiv)     #We calculate how far is our individual to each other individual 
            if g < M:                           #If that other invidual is a neighbour of our current individual
                sum += M - g
        self.grid_crowding_distance = sum

    #GCPD of a given individual. Population is needed for upper and lower boundries of the grid
    #This is a convergence measure
    def calculate_grid_coordinate_point_distance(self, population):
        sum = 0
        for i in range(0, len(self.objectives)):
            d = (population.upper[i]-population.lower[i])/float(population.div)
            if d > 0:   #If d is 0 all individuals have the same value on that objective, thus not contributing to gcpd
                sum += ((self.objectives[i] - (population.lower[i] + self.grid_coordinates[i] * d)) / d)**2
        self.grid_coordinate_point_distance = math.sqrt(sum)


######################################################################
# This class represents an individual which is a Logistic Regression #
######################################################################

class IndividualLR(Individual):

    def __init__(self):
        super().__init__()

    def dominates(self, other_individual):
        super().dominates_standard(other_individual)


###################################################
# Individual Decision Tree for the GrEA algorithm #
###################################################

class IndividualLRGrea(IndividualLR):
    def __init__(self):
        super().__init__()
        self.grid_coordinates = None
        self.grid_rating = None
        self.grid_crowding_distance = None
        self.grid_coordinate_point_distance = None
        self.punishment_degree = 0

    #For grid dominance
    def grid_dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.grid_coordinates, other_individual.grid_coordinates):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

    #Returns grid difference between individuals, having calculated their locations
    def grid_difference(self, other_individual):
        sum = 0
        for k in range(0, len(self.objectives)):
            sum += abs(self.grid_coordinates[k]-other_individual.grid_coordinates[k])
        return sum

    #Grid rating measure for convergence
    #This is a convergence measure
    def calculate_grid_rating(self):
        sum = 0
        for k in range(0, len(self.grid_coordinates)):
            sum += self.grid_coordinates[k]
        self.grid_rating = sum


    #Calculates grid crowding distance (GCD) concerning all the population
    #if the individual belongs to the population itself, at least GCD = M
    #This is a diversity measure
    def calculate_grid_crowding_distance(self, population):
        M = len(self.objectives)
        sum = 0
        for indiv in population:
            g = self.grid_difference(indiv) 
            if g < M:
                sum += M - g
        self.grid_crowding_distance = sum

    #GCPD of a given individual. Population is needed for upper and lower boundries of the grid
    #This is a convergence measure
    def calculate_grid_coordinate_point_distance(self, population):
        sum = 0
        for i in range(0, len(self.objectives)):
            d = (population.upper[i]-population.lower[i])/float(population.div)
            sum += ((self.objectives[i] - (population.lower[i] + self.grid_coordinates[i] * d)) / d)**2
        self.grid_coordinate_point_distance = math.sqrt(sum)