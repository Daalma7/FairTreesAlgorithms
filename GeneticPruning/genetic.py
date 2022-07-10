from hashlib import sha256
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
    
from general import Individual
from general import Individual_NSGA2


class Genetic_Pruning_Process():

    """
    Class that represents a genetic process.
    """

    indiv_class = Individual

    def __init__(self, struc, objs_string, num_gen, num_indiv, prob_cross, prob_mutation):

        """
        Class constructor

        Parameters:
        - struc: tree structure with all calculated metrics, from the general.Tree_Structure class
        - objs_string: Strings defining the objective functions for the optimization process
        - num_gen: Number of generations
        - num_indiv: Number of individuals of the population
        - prob_cross: Probability of crossover
        - prob_mutation: Probability of mutation
        """

        self.struc = struc
        self.objs_string = objs_string
        self.num_gen = num_gen
        self.num_indiv = num_indiv
        self.prob_cross = prob_cross
        self.prob_mutation = prob_mutation
        self.population = []              # We initialize the population to None but it will be directly created
        self.initial_population()           # It directly creates the initial population

    def initial_population(self):
        """
        Creates the initial population.
        """

        children_left = self.struc.clf.tree_.children_left      # Left children of each tree node
        children_right = self.struc.clf.tree_.children_right    # Right children of each tree node
        depth = self.struc.clf.get_depth()                      # Max depth of the tree

        for i in range(self.num_indiv):          # For each individual to be created

            newset = []                 # Representation of the new individual.

            base_denom_prob = depth + np.random.randint(1, int(depth/2))      # Assign base denominator for the probability of pruning (1/denominator)
                                                                              # The minimum value should be at least the tree maximum depth
                                                                              # The additional depth is selected randomly with arbitrary bounds.

            stack = [(0, [], base_denom_prob, 0, False)]  # start with the root node id (0) and its representation, the probability
                                                          # for each of its children to be pruned, the next child to explore, and if the current node must be pruned

            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, repr, cur_denom, next, prun = stack.pop()         # Extract the node id, its representation and odds to be prunned

                # If the left and right child of a node is not the same, we are dealing with a split node (non leaf)
                is_split_node = children_left[node_id] != children_right[node_id]
            
                if is_split_node:                # If a split node
                    if prun:                     # If it is selected to be pruned (if not a split node it should not be selected to be pruned)
                        newset.append(list(repr))
                    else:                        # If it won't be pruned, we will explore its children, lexicographically
                        if next == 0:               # The next child to explore is the left one
                            stack.append((node_id, repr, cur_denom, 1, False)) # With this, we will get lexicographically ordered representations
                            newprun = np.random.rand() < 1 / float(cur_denom)              # We will select the node for a pruning or not
                            stack.append((children_left[node_id], repr + [0], cur_denom-1, 0, newprun))  # Append tree considered and our path based representations.
                        else:                       # The next child to explore is the right one
                            newprun = np.random.rand() < 1 / float(cur_denom)              # We will select the node for a pruning or not
                            stack.append((children_right[node_id], repr + [1], cur_denom-1, 0, newprun)) 
            
            self.population.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newset))    # Create and append the individual

        




    def lex_compare(self, a, b):
        """
        Auxiliary function for lexicographical comparison between 2 arrays

        Parameters:
        - a: First array
        - b: Second array

        Returns:
        - c: Code of which array goest first wrt lexicographical order
            0 if both are equal
            1 if a < b (a is lower)
            2 if a > b (b is lower)
        """

        newa = np.array(a[:min(len(a), len(b))])  # Subarray with the length of the shortests one
        newb = np.array(b[:min(len(a), len(b))])

        try:
            idx = np.where( (newa>newb) != (newa<newb) )[0][0]      # Get the index where both differ. If not exists, returns IndexError

            if a[idx] < b[idx]: return 1
            if a[idx] > b[idx]: return 2

        except IndexError:                                          # If both subarrays are equal, compare the length of the real ones
            
            if len(a) < len(b): return 1
            if len(a) > len(b): return 2
            if len(a) == len(b): return 0

    def crossover(self, indiv1, indiv2):
        """
        Crossover between 2 individuals
        
        Parameters:
        - indiv1: First parent individual
        - indiv2: Second parent individual

        Returns:
        - newindiv1: First child individual
        - newindiv2: Second child individual
        """
        
        # If both individuals represent no prunings, they will be returned
        if len(indiv1.repr) == 0 and len(indiv2.repr) == 0:
            return indiv1, indiv2

        # In any other case
        # We will take the prunings done in each individual, and select randomly to which individual we be assigned.
        repr1 = indiv1.repr.copy()
        repr2 = indiv2.repr.copy()
        i = j = 0
        selected = 0

        newindiv1 = []
        newindiv2 = []

        # We will select a prior probability for each pruning to be assigned to each child.
        # In order to not generate really extreme individuals we will assign a random probability between 0.2 and 0.8
        prob_child_1 = np.random.rand() * 0.6 + 0.2
        
        # First of all, we will select the next pruning to insert into the children. We select it in a lexicographically ordered way
        while i < len(repr1) or j < len(repr2):         # We repeat for all the parents' prunings
            if i == len(repr1):                             # If we no longer have prunings from the first parent
                selected = repr2[j]
                j += 1
            else:
                if j == len(repr2):                         # If we no longer have prunings from the second parent
                    selected = repr1[i]
                    i += 1
                else:                                           # If both parents have prunings yet to be analysed
                    # Compare the next prunings of both parents and select the one which goes first wrt lexicographical order
                    res = self.lex_compare(list(repr1[i]), list(repr2[j]))
                    if res == 2:                # If repr2[j] goes before repr1[i]
                        selected = repr2[j]
                        j += 1
                    else:                       # In any other case
                        selected = repr1[i]
                        i += 1
            
            selected = list(selected)
            # After having chosen it, we will select where will we insert it.
            # We will first create a random value and compare it with the base probability
            if np.random.rand() < prob_child_1:
                # But we will have to check if the are already considered prunings that will make the current pruning invalid.
                # Because of the ordered way they are being inserted, we will only have to compare with the previous one.

                if len(newindiv1) > 0:
                    min1 = min(len(newindiv1[-1]), len(selected))
                    if newindiv1[-1][:min1] == selected[:min1]:
                        # That means that inserting the selected pruning in the first individual will be invalid, in this case we will try to insert it in the second one.
                        # But we should test the same for that second individual
                        if len(newindiv2) > 0:
                            min2 = min(len(newindiv2[-1]), len(selected))
                            if newindiv2[-1][:min2] != selected[:min2]:
                                newindiv2.append(selected)
                        else:
                            newindiv2.append(selected)
                        # In any other case, it will not be appended.
                    else:
                        # In other case we will insert it where it should taking the probability into account.
                        newindiv1.append(selected)
                else:
                    newindiv1.append(selected)
            else:
                # The same for newindiv2
                if len(newindiv2) > 0:
                    min2 = min(len(newindiv2[-1]), len(selected))
                    if newindiv2[-1][:min2] == selected[:min2]:
                        # That means that inserting the selected pruning in the first individual will be invalid, in this case we will insert it in the second one.
                        # But we should test the same for that second individual
                        if len(newindiv1) > 0:
                            min1 = min(len(newindiv1[-1]), len(selected))
                            if newindiv1[-1][:min1] != selected[:min1]:
                                newindiv1.append(selected)
                        else:
                            newindiv1.append(selected)
                        # In any other case, it will not be appended.
                    else:
                        # In other case we will insert it where it should taking the probability into account.
                        newindiv2.append(selected)
                else:
                    newindiv2.append(selected)

        return Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newindiv1), Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newindiv2)


    def tournament(self):
        """
        Applies tournament criterion over population, returning parents population.
        It is currently implemented for single objective functions

        Returns:
        - new_pop: parents population
        """
        
        new_pop = []    # Population of nodes winning the tournament

        longi = len(self.population)
        for i in range(longi):                      # For each individual in the new population
            rand1 = np.random.randint(0, longi)     # Randomly select two distinct individuals
            rand2 = np.random.randint(0, longi)
            while rand1 == rand2:
                rand2 = np.random.randint(0, longi)

            # We select the node with the greates objectives value.
            objectives_1 = self.population[rand1].objectives
            objectives_2 = self.population[rand2].objectives
            if objectives_1 > objectives_2:
                new_pop.append(self.population[rand1])
            else:
                new_pop.append(self.population[rand2])

        return new_pop


    def pop_crossover(self):
        """
        Applies crossover over parents population

        Returns:
        - new_pop: new children population
        """
        
        new_pop = []                    # Children population

        longi = len(self.population)         
        for i in range(int(longi/2)):       # For each pair or parents
            rand = np.random.random()       # We decide if crossover will be done
            if rand < self.prob_cross:           # If so, we apply it

                new_indiv1, new_indiv2 = self.crossover(self.population[2*i], self.population[(2*i)+1])
                new_pop.append(new_indiv1)
                new_pop.append(new_indiv2)
            else:                           # If not, we return the same parents
                new_pop.append(self.population[2*i])
                new_pop.append(self.population[(2*i)+1])

        return new_pop



    def aux_random_value_prob_dict(self, prob_dict, rand):
        """
        Return an element given a probability dictionary, and a random value
        This is an auxiliary function

        Parameters:
        - prob_dict: dictionary with probabilities. The sum of the values has to be 1
        - rand: random value between 0 and 1

        Returns:
        - k: selected key from the dictionary
        """
        total = 0
        
        for k, v in prob_dict.items():
            total += v
            if rand <= total:
                return k


    def mutation(self, indiv):
        """
        Applies mutations randomly over an individual. The mutations may not happen at all, depends of probability

        Parameters:
        - indiv: individual for which the mutation will be done

        Returns:
        - objectives_val: its objectives value
        """

        rand_modify = np.random.random()

        if rand_modify < self.prob_mutation:       # If a mutation will be done
            
            # We will calculate the nodes from which a modification of a pruning can be done            
            leaves = indiv.repr.copy()                  # Considering the prunings which have been applied.
            baselen = len(leaves)

            for leaf in self.struc.base_leaves:     # For each one of the real leaves of the tree
                insert = True
                i = 0
                while i < baselen and insert:                # For each pruning done
                    minlen = min(len(leaves[i]), len(leaf))
                    if(leaves[i][:minlen] == leaf[:minlen]):    # If the pruning goes hierachically before the leaf
                        insert = False
                    i += 1
                
                if insert:
                    leaves.append(leaf)


            # We will now randomly select one of those leaves. We will select it using equal probabilities
            probs = {tuple(leaf): 1/len(leaves) for leaf in leaves}
            #print(leaves)
            #print(probs)
            leaf = list(self.aux_random_value_prob_dict(probs, np.random.random()))         # Select one of the leaves

            # After having selected the leaf, we will now select the mutation applied over that leaf.

            new_repr = indiv.repr.copy()

            if leaf in self.struc.base_leaves:                 # If the leaf is a leaf node of the complete tree, we can only select its parent
                if len(leaf) > 1:                       # It its parent is not the root node
                    # We will insert it in an ordered way
                    l = 0
                    inserted = False
                    while l < len(new_repr) and not inserted:
                        if self.lex_compare(new_repr[l], leaf[:-1]) == 0:
                            del new_repr[l]
                            l = l - 1
                        if self.lex_compare(new_repr[l], leaf[:-1]) == 2:
                            inserted = True
                            new_repr.insert(l, leaf[:-1])
                        l = l + 1
                    
                    if not inserted:                        # It should be inserted in the last position
                        new_repr.append(leaf[:-1])                    # We directly add it to the individual
            else:                                   # In other case we can go up or down in the tree hierarchy.
                new_probs = {}                          # We will create a new space for the possible mutations
                if len(leaf) > 1:
                    new_probs[tuple(leaf[:-1])] = 1/5          # Parent node
                children = self.struc.children_nodes(leaf)
                
                # Children nodes. We have to take into account the possibility of removing the pruning if a children is a leaf node of the actual tree
                # These probabilities have to be modified into more accurate ones
                if len(children) == 2:
                    new_probs[children[0]] = 2/5
                    new_probs[children[1]] = 2/5
                elif len(children) == 1:
                    new_probs['empty'] = 2/5
                    new_probs[children[0]] = 2/5
                else:
                    new_probs['empty'] = 4/5
            
                my_sum = sum(new_probs.values())                                        # Calculate sum for probability normalization
                new_probs = {key: value/my_sum for key, value in new_probs.items()}         # Apply probability normalization

                prun = self.aux_random_value_prob_dict(new_probs, np.random.random())         # Select one of them

                if leaf in new_repr:
                    new_repr.remove(leaf)                                                      # Get rid of the leaf



                if prun != 'empty':                                                     # Add the pruning if necessary
                    l = 0
                    inserted = False
                    while l < len(new_repr) and not inserted:
                        if self.lex_compare(new_repr[l], prun) == 0:
                            del new_repr[l]
                            l = l - 1
                        if self.lex_compare(new_repr[l], prun) == 2:
                            inserted = True
                            new_repr.insert(l, prun)
                        l = l + 1
                    
                    if not inserted:                        # It should be inserted in the last position
                        new_repr.append(prun)                    # We directly add it to the individual

            return Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, list(new_repr))

        return(indiv)

    def genetic_optimization(self, seed):
        """
        Defines the whole optimization process

        Parameters:
        - seed: Random seed

        Returns:
        - self.population: Population after the genetic optimization process
        """

        np.random.seed(seed)

        print("Beggining")
        print(self.population)              # Print initial population ERASE
        for i in range(self.num_gen):       # For each generation
            new_pop = self.tournament()     # We select the parent inidividuals
            new_pop = self.pop_crossover()  # We apply crossover operator
            new_pop = [self.mutation(indiv) for indiv in new_pop] # We apply mutation over individuals
            
            print(i)
            self.population = new_pop       # We update the population to the new created one
        print("End")
        return self.population              # Return the last created population


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class Genetic_Pruning_Process_NSGA2(Genetic_Pruning_Process):

    """
    Class representing a Genetic Pruning Process.

    It inherits all methods and structures from the Genetic_Pruning_Process class. Some of them
    will be redefined

    The individual class from the general file will Individual_NSGA2
    """

    indiv_class = Individual_NSGA2

    def __init__(self, struc, objs, num_gen, num_indiv, prob_cross, prob_mutation):

        """
        Class constructor

        Parameters:
        - struc: tree structure with all calculated metrics, from the general.Tree_Structure class
        - objs: Strings defining the objective functions for the optimization process
        - num_gen: Number of generations
        - num_indiv: Number of individuals of the population
        - prob_cross: Probability of crossover
        - prob_mutation: Probability of mutation
        """

        Genetic_Pruning_Process.__init__(self, struc, objs, num_gen, num_indiv, prob_cross, prob_mutation)

        for i in range(len(self.population)):
            print(self.population[i].repr)
        # Initialization of other variables needed 
        self.fronts = []
        self.domination_count = 0
        self.dominated_solutions = []

    def crowding_distance(self, front):

        """
        Calculates the crowding distance of all individuals belonging to a given front

        Parameters:
        - front: Front for which calculate the crowding distance of each individual
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

    def fast_nondominated_sort(self):

        """
        Fast nondominated sort of the individuals into fronts
        """
        
        self.fronts = [[]]
        for individual in self.population:               # Stablishment of the best front and dominance info for generating the rest of them
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in self.population:
                if individual.dominates(other_individual):                  # If the current solution dominates the other one
                    individual.dominated_solutions.append(other_individual) # It will be added to its list of dominated solutions
                elif other_individual.dominates(individual):                # In any other case
                    individual.domination_count += 1                        # We increase the domination count
            if individual.domination_count == 0:                            # If it is not dominated by any other individual
                individual.rank = 0
                self.fronts[0].append(individual)
        i = 0
        while len(self.fronts[i]) > 0:        # While there are more individuals within the current front
            temp = []
            for individual in self.fronts[i]:
                for other_individual in individual.dominated_solutions:     # For each individual dominated by the current solution
                    other_individual.domination_count -= 1                  # We subtract 1 to the current counter 
                    if other_individual.domination_count == 0:              # If it reaches 0, there are no other individuals which dominate it, so it belongs to the next front
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            self.fronts.append(temp)

    def tournament(self):
        """
        Applies tournament criterion over population, returning parents population.

        Returns:
        - new_pop: parents population
        """

        self.fast_nondominated_sort()
        for front in self.fronts:
            self.crowding_distance(front)
        
        new_pop = []    # Population of nodes winning the tournament

        longi = len(self.population)
        for i in range(longi):                      # For each individual in the new population
            rand1 = np.random.randint(0, longi)     # Randomly select two distinct individuals
            rand2 = np.random.randint(0, longi)
            while rand1 == rand2:
                rand2 = np.random.randint(0, longi)

            # We select the node with the best rank. It there is none, we select the one with the best crowding distance
            best = None
            if self.population[rand1].rank < self.population[rand2].rank:
                best = self.population[rand1]
            elif self.population[rand1].rank > self.population[rand2].rank:
                best = self.population[rand2]
            else:
                if self.population[rand1].crowding_distance > self.population[rand2].crowding_distance:
                    best = rand1
                else:
                    best = rand2
            new_pop.append(best)

        return new_pop

    
    def genetic_optimization(self, seed):
        """
        Defines the whole optimization process
        """

        np.random.seed(seed)


        print("Start")
        for i in range(len(self.population)):
            print(self.population[i].repr)
        for i in range(self.num_gen):
            new_pop = self.tournament()
            new_pop = self.pop_crossover()
            new_pop = [self.mutation(indiv) for indiv in new_pop]
            
            print(i)
            self.population = new_pop
        print("End")

        self.fast_nondominated_sort()
        return self.fronts[0]                       # Returns the best individuals

