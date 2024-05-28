from hashlib import sha256
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
    
from individual import Individual
from individual import Individual_NSGA2
from sympy import symbols, nsolve
import random
import copy
import math
import pandas as pd
from ml import create_gen_stats_df, update_gen_stats_df, create_gen_population_df, update_gen_population
from joblib import Parallel, delayed

import time

class Genetic_Pruning_Process():
    """
    Class that repreesents a genetic process.
    """

    indiv_class = Individual

    def __init__(self, struc, objs_string, num_gen, num_indiv, prob_cross, prob_mutation):
        """
        Class constructor
            Parameters:
                - struc: Tree structure with all calculated metrics, from the individual.Tree_Structure class
                - objs_string: Strings defining our objective functions for the optimization process
                - num_gen: Number of generations
                - num_indiv: Number of individuals of our population
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


    # MODIFIABLE
    def initial_population(self):
        """
        Creates the initial population.
        In this case, it is equally likely to prune each node at equal depth.
        """

        children_left = self.struc.clf.tree_.children_left      # Left children of each tree node
        children_right = self.struc.clf.tree_.children_right    # Right children of each tree node
        depth = self.struc.clf.get_depth()                      # Max depth of the tree
        
        x=symbols('x')

        base_prob = nsolve(x - (1-x)**(depth-1), x, 0) /2
        assert(nsolve(x - (1-x)**(depth-1), x, 0) > base_prob)
        assert(base_prob > 0)
        

        for i in range(self.num_indiv):          # For each individual to be created

            newset = []                 # representation of the new individual.
            stack = [(0, [], 0, 0, False)]  # start with the root node id (0) and its repreesentation, the probability
                                                          # for each of its children to be pruned, the next child to explore, and if the current node must be pruned

            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, repre, cur_depth, next, prun = stack.pop()         # Extract the node id, its repreesentation and odds to be prunned

                # If the left and right child of a node is not the same, we are dealing with a split node (non leaf)
                is_split_node = children_left[node_id] != children_right[node_id]
            
                if is_split_node:                # If a split node
                    if prun:                     # If it is selected to be pruned (if not a split node it should not be selected to be pruned)
                        newset.append(list(repre))
                    else:                        # If it won't be pruned, we will explore its children, lexicographically
                        if next == 0:               # The next child to explore is the left one
                            stack.append((node_id, repre, cur_depth, 1, False)) # With this, we will get lexicographically ordered repreesentations
                            newprun = np.random.rand() < base_prob / (1-base_prob)**cur_depth              # We will select the node for a pruning or not
                            stack.append((children_left[node_id], repre + [0], cur_depth+1, 0, newprun))  # Append tree considered and our path based repreesentations.
                        else:                       # The next child to explore is the right one
                            newprun = np.random.rand() < base_prob / (1-base_prob)**cur_depth              # We will select the node for a pruning or not
                            stack.append((children_right[node_id], repre + [1], cur_depth+1, 0, newprun)) 
            
            self.population.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newset, 'initialization'))    # Create and append the individual
            #print(newset)


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
            idx = np.where( (newa>newb) != (newa<newb) )[0][0]      # Get the index where both differ. If it not exists, returns IndexError
            if a[idx] < b[idx]: return 1
            if a[idx] > b[idx]: return 2

        except IndexError:                                          # If both subarrays are equal, compare the length of the real ones
            if len(a) < len(b): return 1
            if len(a) > len(b): return 2
            if len(a) == len(b): return 0


    # MODIFIABLE
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
        if len(indiv1.repre) == 0 and len(indiv2.repre) == 0:
            return indiv1, indiv2

        # In any other case
        # We will take the prunings done in each individual, and select randomly to which individual we be assigned.
        repre1 = indiv1.repre.copy()
        repre2 = indiv2.repre.copy()
        i = j = 0
        selected = 0

        newindiv1 = []
        newindiv2 = []

        # We will select a prior probability for each pruning to be assigned to each child.
        # In order to not generate really extreme individuals we will assign a random probability between 0.2 and 0.8
        prob_child_1 = np.random.rand() * 0.6 + 0.2

        # First of all, we will select the next pruning to insert into the children. We select it in a lexicographically ordered way
        while i < len(repre1) or j < len(repre2):         # We repeat for all the parents' prunings
            if i == len(repre1):                             # If we no longer have prunings from the first parent
                selected = repre2[j]
                j += 1
            else:
                if j == len(repre2):                         # If we no longer have prunings from the second parent
                    selected = repre1[i]
                    i += 1
                else:                                           # If both parents have prunings yet to be analysed
                    # Compare the next prunings of both parents and select the one which goes first wrt lexicographical order
                    res = self.lex_compare(list(repre1[i]), list(repre2[j]))
                    if res == 2:                # If repre2[j] goes before repre1[i]
                        selected = repre2[j]
                        j += 1
                    else:                       # In any other case
                        selected = repre1[i]
                        i += 1
            
            selected = list(selected)
            # After having chosen it, we will select where will we insert it.
            # We will first create a random value and compare it with the base probability
            if np.random.rand() < prob_child_1:
                # But we will have to check if the are already considered prunings that will make the current pruning invalid.
                # Because of the ordered way they are being inserted, we will only have to compare with the previous one.

                
                if len(newindiv1) > 0:                                          # In case there already is a pruning inserted
                    min1 = min(len(newindiv1[-1]), len(selected))               # We obtain the common part with the last pruning.
                    if newindiv1[-1][:min1] == selected[:min1]:
                        # That means that inserting the selected pruning in the first individual will be invalid, in this case we will try to insert it in the second one.
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
        
        # print("aaaaaaa")
        # print(dict_newindiv1)
        # print(dict_newindiv2)
        # print("aaaaaaa")
        dict_newindiv1 = {tuple(k):1 for k in newindiv1}
        dict_newindiv2 = {tuple(k):1 for k in newindiv2}
        assert(len(newindiv1) == len(dict_newindiv1))
        assert(len(newindiv2) == len(dict_newindiv2))


        return Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newindiv1, 'crossover'), Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, newindiv2, 'crossover')

    # Overwritten
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


    # TODO: Parallelize
    def parallel_pop_crossover(self, parent_1, parent_2):
        rand = np.random.random()       # We decide if crossover will be done
        n_p = []
        if rand < self.prob_cross:           # If so, we apply it
            new_indiv1, new_indiv2 = self.crossover(parent_1, parent_2)
            n_p.append(new_indiv1)
            n_p.append(new_indiv2)
        else:                           # If not, we return the same parents
            n_p.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, parent_1.repre, parent_1.creation_mode, parent_1.objectives))
            n_p.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, parent_2.repre, parent_2.creation_mode, parent_2.objectives))
        return n_p
    

    # MODIFIABLE
    def pop_crossover(self, parents, parallel=False):
        """
        Applies crossover over parents population
            Returns:
                - new_pop: new children population
        """
        new_pop = []                    # Children population
        longi = len(parents)
        if parallel:
            new_pop = Parallel(n_jobs=-1)(delayed(self.parallel_pop_crossover)(self.population[2*i], self.population[(2*i)+1]) for i in range(int(longi/2)))
            new_pop = [child for children in new_pop for child in children]
        else:
            for i in range(int(longi/2)):       # For each pair or parents
                rand = np.random.random()       # We decide if crossover will be done
                if rand < self.prob_cross:           # If so, we apply it
                    new_indiv1, new_indiv2 = self.crossover(parents[2*i], parents[(2*i)+1])
                    new_pop.append(new_indiv1)
                    new_pop.append(new_indiv2)
                else:                           # If not, we return the same parents
                    new_pop.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, self.population[2*i].repre, self.population[2*i].creation_mode, self.population[2*i].objectives))
                    new_pop.append(Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, self.population[(2*i)+1].repre, self.population[(2*i)+1].creation_mode, self.population[(2*i)+1].objectives))

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

    # TODO: Revisar
    # MODIFIABLE
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
            #print("Begin Mutation")
            # We will calculate the nodes from which a modification of a pruning can be done 
            # We will insert them in an ORDERED way
            leaves = indiv.repre.copy()                  # Considering the prunings which have been applied.
            if len(leaves) > 0:
                # We will now insert all the actual leaves not pruned by any already considered pruning into the representation
                j = len(self.struc.base_leaves) - 1
                i = 0
                end = False
                prev_prun = False
                while j > -1:               # We traverse backwards as they were inserted from right to left
                    #print("-")
                    #print(self.struc.base_leaves[j])
                    #print(leaves[i])
                    minlen = min(len(self.struc.base_leaves[j]), len(leaves[i]))     # Check if coincident (pruned leaf)
                    if(leaves[i][:minlen] == self.struc.base_leaves[j][:minlen]):    # If the pruning goes hierachically before the leaf
                        prev_prun = True                    # We will not include that leaf
                    else:                                       #   In other case
                        if prev_prun:                       # It the operation before was not to include, we have to check the next pruning (if exists)
                            j = j + 1
                            if i < len(leaves)-1:           # Check if we have already checked all prunings.
                                i = i + 1
                            else:                       # If that is the case, we flag it, for future leaves insertion
                                end = True
                        else:                               # In other case, we include the leaf
                            if not end:                         # In case we have not reached the end of leaves 
                                leaves.insert(i, self.struc.base_leaves[j])
                                i = i+1
                            else:                           # If we've reached the end, we insert them at the very end
                                leaves.append(self.struc.base_leaves[j])
                        prev_prun = False                   # Previous operation was not to prune
                    j = j - 1
            else:
                # All leaves will appear
                j = len(self.struc.base_leaves) - 1
                while j > -1:
                    leaves.append(self.struc.base_leaves[j])
                    j = j - 1
            # We will now randomly select one of those leaves. We will select it using equal probabilities
            new_repre = indiv.repre.copy()

            # Select random leaf
            leaf = copy.deepcopy(random.choice(leaves))
            move_distance = random.randint(1, math.ceilself.struc.clf.tree_.max_depth / 10)
            # After having selected the leaf, we will now select the mutation applied over that leaf.
            if random.random() > 0.5:
                # Move up
                if len(leaf) > move_distance:                       # If we can traverse up the amount specified in move_distance
                    leaf = leaf[:-move_distance]
                else:                                               # In any other case, we have to move up to the root node
                    leaf = []
            else:
                # Move down
                for i in range(move_distance):                      # We will try to move down the number speficied in move_distance
                    if random.random() > 0.5:                           # Create the random path
                        leaf.append(1)
                    else:
                        leaf.append(0)
                
                is_prun = leaf in self.struc.pruning_space
                while not is_prun:
                    leaf = leaf[:-1]
                    is_prun = leaf in self.struc.pruning_space
            
            # We will now insert it
            if not leaf in new_repre:
                # Once we've decided that we will add the new calculated leaf 
                inserted = False
                if len(leaf) > 1:                       # It its parent is not the root node
                    # We will insert it in an ordered way
                    l = 0
                    end = False
                    while l < len(new_repre) and not inserted:
                        minlen = min(len(new_repre[l]), len(leaf))
                        res = self.lex_compare(new_repre[l][:minlen], leaf[:minlen])
                        if res == 0:                    # We need to prune that branch
                            del new_repre[l]
                            l = l - 1
                        if res == 2:                    # We insert the pruning just before
                            inserted = True
                            new_repre.insert(l, leaf)
                
                        l = l+1
                    
                    if not inserted:
                        new_repre.append(leaf)

            """
            # for leaf in leaves:
                #probs = {tuple(leaf): 1/len(leaves) for leaf in leaves}
                #leaf = list(self.aux_random_value_prob_dict(probs, np.random.random()))         # Select one of the leaves
    
            # After having selected the leaf, we will now select the mutation applied over that leaf.
            if leaf in self.struc.base_leaves:                 # If the leaf is a leaf node of the complete tree, we can only select its parent, or leave it (equally likely)
                if random.random() > 0.5:                            # Equally likely to take it or not
                    inserted = False
                    if len(leaf) > 1:                       # It its parent is not the root node
                        # We will insert it in an ordered way
                        l = 0
                        end = False
                        while l < len(new_repre) and not inserted:
                            minlen = min(len(new_repre[l]), len(leaf[:-1]))
                            res = self.lex_compare(new_repre[l][:minlen], leaf[:minlen])
                            if res == 0:                    # We need to prune that branch
                                del new_repre[l]
                                l = l - 1
                            if res == 2:                    # We insert the pruning just before
                                inserted = True
                                new_repre.insert(l, leaf[:-1])                        
                            l = l+1
                        
                        if not inserted:
                            new_repre.append(leaf[:-1])


            
                    if leaf in new_repre:
                        new_repre.remove(leaf)                                                      # Get rid of the leaf
            
                    if not prun == 'empty':
                        # Once we've decided that we will add the new calculated leaf 
                        inserted = False
                        if len(prun) > 1:                       # It its parent is not the root node
                            # We will insert it in an ordered way
                            l = 0
                            end = False
                            while l < len(new_repre) and not inserted:
                                minlen = min(len(new_repre[l]), len(prun))
                                res = self.lex_compare(new_repre[l][:minlen], prun[:minlen])
                                if res == 0:                    # We need to prune that branch
                                    del new_repre[l]
                                    l = l - 1
                                if res == 2:                    # We insert the pruning just before
                                    inserted = True
                                    new_repre.insert(l, prun)
                        
                                l = l+1
                            
                            if not inserted:
                                new_repre.append(prun)

            else:                                   # In other case we can go up or down in the tree hierarchy.
                new_probs = {tuple(leaf): 1}                          # We will create a new space for the possible mutations (keeping )
                if len(leaf) > 1:
                    new_probs[tuple(leaf[:-1])] = 1/2          # Parent node
                children = self.struc.children_nodes(list(leaf))
                
                # Children nodes. We have to take into account the possibility of removing the pruning if a children is a leaf node of the actual tree
                # These probabilities have to be modified into more accurate ones
                if len(children) == 2:
                    new_probs[children[0]] = 1/2
                    new_probs[children[1]] = 1/2
                elif len(children) == 1:
                    new_probs['empty'] = 1/2
                    new_probs[children[0]] = 1/2
                else:
                    new_probs['empty'] = 1/2
            
                my_sum = sum(new_probs.values())                                        # Calculate sum for probability normalization
                new_probs = {key: value/my_sum for key, value in new_probs.items()}         # Apply probability normalization

                prun = self.aux_random_value_prob_dict(new_probs, np.random.random())         # Select one of them
                
                if not prun is leaf:
                    if leaf in new_repre:
                        new_repre.remove(leaf)                                                      # Get rid of the leaf
            
                    if not prun == 'empty':
                        # Once we've decided that we will add the new calculated leaf 
                        inserted = False
                        if len(prun) > 1:                       # It its parent is not the root node
                            # We will insert it in an ordered way
                            l = 0
                            end = False
                            while l < len(new_repre) and not inserted:
                                minlen = min(len(new_repre[l]), len(prun))
                                res = self.lex_compare(new_repre[l][:minlen], prun[:minlen])
                                if res == 0:                    # We need to prune that branch
                                    del new_repre[l]
                                    l = l - 1
                                if res == 2:                    # We insert the pruning just before
                                    inserted = True
                                    new_repre.insert(l, prun)
                        
                                l = l+1
                            
                            if not inserted:
                                new_repre.append(prun)
            """         

            return Genetic_Pruning_Process.indiv_class(self.struc, self.objs_string, list(new_repre), 'mutation')

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
        #print(self.population)              # Print initial population ERASE
        for i in range(self.num_gen):       # For each generation
            parents = self.tournament()     # We select the parent inidividuals
            children = self.pop_crossover(parents)  # We apply crossover operator
            # TODO: Parallelize
            children = [self.mutation(indiv) for indiv in children] # We apply mutation over individuals
            
            #print(i)
            self.population = children       # We update the population to the new created one
        print("End")
        return self.population              # Return the last created population


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class Genetic_Pruning_Process_NSGA2(Genetic_Pruning_Process):

    """
    Class repreesenting a Genetic Pruning Process.
    It inherits all methods and structures from the Genetic_Pruning_Process class. Some of them
    will be redefined.
    The individual class from the general file will Individual_NSGA2.
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

        #for i in range(len(self.population)):   
            #print(self.population[i].repre)
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
                    best = self.population[rand1]
                else:
                    best = self.population[rand2]
            new_pop.append(best)

        return new_pop

    def genetic_optimization(self, seed, store=True):
        """
        Defines the whole optimization process
        """
        np.random.seed(seed)

        start_p_time = time.process_time()
        start_t_time = time.time()

        gen_stats_df = create_gen_stats_df()
        gen_population_df = create_gen_population_df(self.objs_string, seed)
        
        print("Start")
        #for i in range(len(self.population)):
            #print(self.population[i].repre)
        

        for i in range(self.num_gen):
            print(f"{i}")

            # Store data
            gen_stats_df = update_gen_stats_df(gen_stats_df, self.population, time.process_time() - start_p_time, time.time() - start_t_time)
            gen_population_df = update_gen_population(gen_population_df, self.population, self.objs_string, seed)
            start_p_time = time.process_time()
            start_t_time = time.time()

            # NSGA-II Process
            parents = self.tournament()                             # Tournament
            children = self.pop_crossover(parents)                   # Crossover
            #list_mutate = [True if np.random.random() < self.prob_mutation else False for x in range(len(children))]
            #print(list_mutate)
            children = [self.mutation(indiv) for indiv in children]   # Mutation
            self.population.extend(children)                         # Selection
            self.fast_nondominated_sort()

            new_population = []
            front_num = 0
            while len(new_population) + len(self.fronts[front_num]) <= self.num_indiv:
                #print(new_population, len(self.fronts[front_num]))
                self.crowding_distance(self.fronts[front_num])
                new_population.extend(self.fronts[front_num])
                front_num += 1
            self.crowding_distance(self.fronts[front_num])
            self.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.fronts[front_num][0:self.num_indiv-len(new_population)])
          
            # New generation
            self.population = new_population






        self.fast_nondominated_sort()
        print("End")

        return self.fronts[0], gen_stats_df, gen_population_df
        #return self.fronts[0]                       # Returns the best individuals