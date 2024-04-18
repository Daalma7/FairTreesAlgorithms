import math

class Population:
    """
    Class representing a population of individuals
    """
    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        """
        Extend the population with new individuals
            Parameters:
                - new_individuals: New individuals to extend the population
        """
        self.population.extend(new_individuals)

    def append(self, new_individual):
        """
        Append the population with a new individual
            Parameters:
                - new_individuals: New individual to append to the population
        """
        self.population.append(new_individual)


class PopulationGrea(Population):
    """
    Class reprsenting a population of individuals, using the GrEA algorithm
    """
    def __init__(self, div):
        super().__init__()
        self.upper = [] #Upper boundries for each grid dimension
        self.lower = [] #Lower boundries for each grid dimension
        self.div = div  #Number of divisions to make in each dimension of the grid
    
    def calculate_grid_boundries(self):
        """
        Calculate grid boundries for that population
        Has to be invoked after every population selection or initialization
            Returns:
                - grid boundries for that population
        """
        self.upper = []
        self.lower = []
        for i in range(0, len(self.population[0].objectives)):
            max = 0
            min = float('inf')
            for element in self.population:     #Calculate minimum and maximum value on each objectivw
                if element.objectives[i] > max:
                    max = element.objectives[i]
                if element.objectives[i] < min:
                    min = element.objectives[i]
            self.upper.append(min - (max - min) / (2 * self.div))   #Then apply following formulas
            self.lower.append(max + (max - min) / (2 * self.div))

    def calculate_grid_locations(self):
        """
        Calculates and updates grid location of all population individuals
        """
        for indiv in self.population:
            loc = []
            for k in range(0, len(self.upper)):
                d = float(self.upper[k]-self.lower[k])/self.div
                if d == 0:      #As all individuals have exactly the same value on that obj so that upper and lower is the same
                    loc.append(0)
                else:
                    loc.append(math.floor(float(indiv.objectives[k]-self.lower[k])/d))
            indiv.grid_coordinates = loc
    

    def findout_best(self):
        """
        Finds the best solution of the current population, having been initialized the
        quality measures of all its individuals
        """
        best = self.population[0]
        for i in range(1, len(self.population)):
            if self.population[i].grid_rating < best.grid_rating:
                best = self.population[i]
            else:
                if self.population[i].grid_rating == best.grid_rating:
                    if self.population[i].grid_crowding_distance < best.grid_crowding_distance:
                        best = self.population[i]
                    else:
                        if self.population[i].grid_crowding_distance == best.grid_crowding_distance:
                            if self.population[i].grid_coordinate_point_distance < best.grid_coordinate_point_distance:
                                best = self.population[i]
        return best

    def GR_adjustment(self,q):
        """
        Adjust the grid rating of the individuals using a new population.
        """
        E = [] #Individuals in P which grid difference to q is 0
        N = [] #Individuals in P which grid difference to q is < nº of objectives (neighbours)
        G = [] #Individuals in P which are grid-dominated by q
        NG = [] #Individuals in P which aren't grid-dominated by q
        M = len(self.population[0].objectives) #Number of objectives

        #First we need to calculate all those sets
        for p in self.population:
            d = q.grid_difference(p)
            if d < M:
                N.append(p)
                if d == 0:
                    E.append(p)
            if q.grid_dominates(p):
                G.append(p)
            else:
                NG.append(p)
        
        #Once calculated, we will update them as considered in the paper, which is a good approach
        for p in E:
            p.grid_rating = p.grid_rating + M + 2
        for p in G:
            p.grid_rating = p.grid_rating + M
        for p in NG:
            if p not in E:
                p.punishment_degree = 0
        for p in N:
            if p in NG and p not in E:
                d = M - p.grid_difference(q)
                if p.punishment_degree < d:
                    p.punishment_degree = d
                    #now we need to calculate G(p)
                    G_p = []
                    for w in self.population:
                        if p.grid_dominates(w):
                            G_p.append(w)
                    for r in G_p:
                        if r not in G and r not in E:
                            if r.punishment_degree < p.punishment_degree:
                                r.punishment_degree = p.punishment_degree
        for p in NG:
            if p not in E:
                p.grid_rating = p.grid_rating + p.punishment_degree