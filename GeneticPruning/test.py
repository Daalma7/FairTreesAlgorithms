from hashlib import sha256
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


import graphviz
from sklearn.tree._tree import TREE_LEAF



wc_data = pd.read_csv("wdbc.data", header=None)
# Generamos las etiquetas
wc_target = wc_data.iloc[:,1].replace(["B","M"], [0,1])
# Generamos los predictores
wc_data = wc_data.iloc[:,2:]
# Vamos a binarizar el primer atributo y lo vamos a considerar como protegido

wc_data = wc_data.iloc[:,2:]
# Vamos a binarizar el primer atributo y lo vamos a considerar como protegido
prot = wc_data.iloc[:,0]
mean = prot.mean()
print(mean)
wc_data.iloc[:,0] = np.where(prot < mean, 0, 1)
prot = wc_data.iloc[:,0]
print(wc_data.shape)

print(prot)
print(wc_data.shape[0]-sum(prot), sum(prot))

# Leemos los datos de iris
clf = DecisionTreeClassifier(random_state=0)

clf.fit(wc_data.to_numpy(), wc_target.to_numpy())

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
#graph.render("iris") 

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

# TODO comentar y realizar correctamente
class Tree_Structure:
    """
    EXPLICAR

    Parameters:
    - root: Root node of the tree
    - children: List of children nodes

    Methods:
    - 
    -

    """

    def __init__(self, data, prot, y, clf):
        self.data = data
        self.prot = prot
        self.y = y
        self.clf = clf
        self.fair_dict, self.total_samples_dict, self.base_leaves = self.__aux_fitness_metrics()
        self.pruning_space = self.__calc_pruning_space()

    def __aux_fitness_metrics(self):
        """
        This function calculates different structures for the base classifier in order to be more
        efficient calculating each individual fitness metrics
        
        Parameters:
        - clf: Sklearn binary classification tree from which the prunings will be made
        - prot: Protected attribute
        - y: Target attribute

        Returns:
        - fair_dict: Dictionary indicating well classified examples inside that node
            - keys: set representing the node using its decision path
            - values: list of lists of integers indicating the class and the protected attribute of the node.
                - First dimension = class
                - Second dimension = protected attribute

        - total_samples_dict: Dictionary indicating total amount of training examples that fall into that node
            - keys: set representing the node using its decision path
            - values: integer indicating the total amount of training examples that fall in that node
        
        - base_leaves: Array containing the representation of the leave nodes that the base tree has 
        """
        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        values = self.clf.tree_.value

        total_samples_dict = {}
        base_leaves = []    # Array to add the leaves of the base tree
        fair_dict = {}

        assoc_dict = {}         # Association dictionary to relate node_ids with our current representation

        stack = [(0, [])]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, repr = stack.pop()         # Extract the node id and its representation
            assoc_dict[node_id] = repr          # Create the representation for that node.

            total_samples_dict[tuple(repr)] = np.sum(values[node_id])   # First of all we calculate the total amount of training samples in that node

            template = []                               # We create a list of lists for the fair dict dictionary
            for i in range(pd.unique(self.y).shape[0]):      # First dimension represents the class
                template.append([])
                for j in range(pd.unique(self.prot).shape[0]):   # Second dimension represents the protected attribute
                    template[i].append(0)
            fair_dict[tuple(repr)] = template        # We assign the created object to the node.
            # It is importat to create one distinct list of lists to every single node.

            # If the left and right child of a node is not the same, we are dealing with a split node (non leaf)
            is_split_node = children_left[node_id] != children_right[node_id]
        
            if is_split_node:                # If a split node, append left and right children and depth to `stack` so we can loop through them
                stack.append((children_left[node_id], repr + [0]))  # Append tree considered and our path based representations.
                stack.append((children_right[node_id], repr + [1]))
            else:                           # If a leaf node, append the node id to `base_leaves`
                base_leaves.append(tuple(repr))  # Append our representation to the list of leaves to return
        
        # We will now create the fairness structure
            # First dimension for specifying the class
            # Second dimension for specifying the protected attribute
        
        # We will update each node in the fair structure with the information of the training data
        node_indicator = self.clf.decision_path(self.data)             # First of all we obtain the decision path for each training example
        leaf_id = self.clf.apply(self.data)                            # And we compute where each specific instance falls.
        for sample_id in range(self.data.shape[0]):               # For each training example
            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]

            # We update the fair structure
            for node_id in node_index:
                # end if the node is a leaf node
                if leaf_id[sample_id] == node_id:
                    fair_dict[tuple(assoc_dict[node_id])][self.y[sample_id]][self.prot[sample_id]] += 1
                    continue
                fair_dict[tuple(assoc_dict[node_id])][self.y[sample_id]][self.prot[sample_id]] += 1

        return fair_dict, total_samples_dict, set(base_leaves)
    

    def __calc_pruning_space(self):
        """
        This function defines the pruning (search) space. 
        
        Parameters:
        - clf: Sklearn binary classification tree from which the prunings will be made

        Returns:
        - repr_space: Tuple containing all possible node codes from which the prunings
                    can be done
        """

        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        repr_space = []
        stack = [(0, [])]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, repr = stack.pop()

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], repr + [0]))
                stack.append((children_right[node_id], repr + [1]))
                repr_space.append(tuple(repr))
        
        return repr_space[1:]
    
    def space_probs(self):
        """
        Defines the probability of each node for being selected in each initial population
        individual. It is calculated inversely proportional to each node depth, and depends
        to the actual depth of the tree. The maximium probability is initialy fixed to 0.5,
        which will be assigned to the maximum depth leaves. The value decreases by negative
        powers of 2. This means that 2 similar trees with very different maximum depth will
        have very different associated probabilities.

        Parameters:
        - space: Tuple containing all nodes where prunings might be done

        Returns:
        - dict_prob: Dictionary containing the probability for each node of being selected
        """

        maxlen = 0
        dict_prob = {}
        for a in self.pruning_space:         # Calculation of the maximum depth of the tree.
            if len(a) > maxlen:
                maxlen = len(a)
            dict_prob[a] = len(a)
        
        # We will now calculate the probability of each node, given the maximum depth
        dict_prob = {a: 1/(2**(maxlen-v+1)) for a, v in dict_prob.items()}
        return dict_prob

    def node_accuracy(self, repr):
        """
        Calculates the accuracy of a given node.

        Parameters:
        - repr: Representation of the node
        - fair_dict: Fairness structure

        Returns:
        - accuracy: Accuracy of the node
        """

        acc = []
        for elem in self.fair_dict[repr]:
            acc.append(sum(elem))

        return max(acc) / self.number_of_indivs(repr)


    def number_of_indivs(self, repr):
        """
        Calculates the number of individuals in a given node.

        Parameters:
        - repr: Representation of the node
        - fair_dict: Fairness structure

        Returns:
        - num: Amount of individuals in the node
        """

        num = 0
        for elem in self.fair_dict[repr]:
            for elem2 in elem:
                num += elem2
        
        return num

    def apply_pruning(self, repr):
        """
        Apply a given pruning to the base clasifier

        Parameters:
        - repr: representation of the pruning

        Returns:
        - new_clf: prunned tree clasifier
        """
        new_clf = copy.deepcopy(clf)

        # For each pruning to do:
        for prun in repr:
            # We will convert the pruning code into the node in which the pruning will be done
            cur_node_index = 0
            for i in range(len(prun)):          # For each node in the node path, we will traverse it to get to a leaf
                if prun[i] == 0:
                    cur_node_index = new_clf.tree_.children_left[cur_node_index]
                else:
                    cur_node_index = new_clf.tree_.children_right[cur_node_index]
            
            # When we get to a leaf, we will prune it, removing the children of that node
            new_clf.tree_.children_left[cur_node_index] = TREE_LEAF
            new_clf.tree_.children_right[cur_node_index] = TREE_LEAF
            
        return new_clf
    
    def children_nodes(self, sing_prun):
        """
        Returns which are the children nodes from a pruning space, if they exist.

        Parameters:
        - pruning: the pruning code from which calculate the children nodes
        - space: the total children space

        Returns:
        - children: the children codes of the pruning, if any.
        """

        children = []
        pos_0 = sing_prun + (0,)          # Codes of the possible children nodes
        pos_1 = sing_prun + (1,)
        if pos_0 in self.pruning_space:              # If the children node exists in the space of possible prunings, we return them
            children.append(pos_0)
        if pos_1 in self.pruning_space:
            children.append(pos_1)
        
        return children

    def correct_indiv(self, repr):
        """
        Return correct individuals. The values conforming an individual may have inconsistencies
        due to redundant prunings done in already prunned branches. This function corrects it.
        
        Parameters:
        - indiv: Set representing an individual to which correct its representation

        Returns:
        - indiv: Corrected individual
        """
        
        if len(repr) > 0:                              # If at least 1 pruning is done
            repr = list(repr)                         # Consider all prunings
            new_repr = []                              # New correct individual to be created
            add_list = np.zeros(len(repr)).tolist()    # Binary addition list

            # Calculation of hierarchical covered prunings
            for i in range(len(repr)):                 # For each pruning
                for j in range(i+1, len(repr)):        # For each pruning after it
                    if repr[i] == repr[j][:len(repr[i])]:    # Calculate if the first pruning covers the second
                        add_list[j] = 1                 # If so, mark the second to not be included
                    elif repr[j] == repr[i][:len(repr[j])]:  # Calculate if the second pruning covers the first
                        add_list[i] = 1
            
            # Cleaning of those prunings
            for i in range(len(repr)):
                if add_list[i] < 1:             # If the pruning is not covered by another
                    new_repr.append(repr[i])  # Add it to the new individual
            repr = new_repr
        
        return set(repr)
    











# TODO comentar y realizar correctamente
class Individual:
    """
    Class that represents an individual.

    Parameters:
    - repr: Representation of the individual
    - fitness: Fitness value of the individual
    """

    def __init__(self, struc, repr):
        self.struc = struc
        self.repr = self.struc.correct_indiv(repr)   # Each individual created will be automatically corrected in order to ensure consistency
        self.fitness = self.__fitness()     # And also its fitness value will be calculated


    def __fitness(self):
        """
        Calculates the fitness value of a certain individual. For doing so, the actual
        pruning is calculated over the base classifier and its results are used.

        Parameters:
        - indiv: individual for which the fitness will be calculated
        - base_leaves: Representation of the leaves of the tree
        - fair_dict: Fairness structure
        - total_samples_dict: Dictionary indicating total amount of training examples that fall into each node

        Returns:
        - fitness_val: its fitness value
        """
        
        # First of all, we will calculate the real leaf nodes of the individual, given
        # the prunings that will be applied.

        leaf_nodes = self.struc.base_leaves.copy()
        for elem in self.repr:
            leaf_nodes.add(elem)
        leaf_nodes = self.struc.correct_indiv(leaf_nodes)

        # We will now calculate the accuracy of this individual knowing the leaf nodes
        acc = []
        num_indivs = []
        for elem in leaf_nodes:
            num = self.struc.number_of_indivs(elem)         
            acc.append(self.struc.node_accuracy(elem) * num)    # Accuracy of the leaf node
            num_indivs.append(num)                              # Number of individuals in the leaf node
        
        # We will now calculate the fairness of the individual
        # FPR

        # First of all we will build the general confusion matrix divided by protected attribute.
            # First dimension: Actual class
            # Second dimension: Predicted class
        
        conf_mat_0 = [[0,0], [0,0]]
        conf_mat_1 = [[0,0], [0,0]]

        for elem in leaf_nodes:
            cur_fair_dict = self.struc.fair_dict[elem]
            if sum(cur_fair_dict[0]) > sum(cur_fair_dict[1]):
                conf_mat_0[0][0] += cur_fair_dict[0][0]
                conf_mat_0[1][0] += cur_fair_dict[1][0]
                conf_mat_1[0][0] += cur_fair_dict[0][1]
                conf_mat_1[1][0] += cur_fair_dict[1][1]
            else:
                conf_mat_0[0][1] += cur_fair_dict[0][0]
                conf_mat_0[1][1] += cur_fair_dict[1][0]
                conf_mat_1[0][1] += cur_fair_dict[0][1]
                conf_mat_1[1][1] += cur_fair_dict[1][1]

        
        fpr_0 = conf_mat_0[0][1] / (conf_mat_0[0][1] + conf_mat_0[0][0])
        fpr_1 = conf_mat_1[0][1] / (conf_mat_1[0][1] + conf_mat_1[0][0])


        return np.sum(np.array(acc) / sum(num_indivs)), 1-abs(fpr_0-fpr_1)         # Total accuracy


    
    

    







# TODO: Comentar y explicar
class Genetic_Pruning_Process():
    """
    Class that represents a genetic process.

    Parameters:
    - struc: Structure of the problem
    - population: Population of the genetic process
    - base_leaves: Representation of the leaves of the tree
    - fair_dict: Fairness structure
    - total_samples_dict: Dictionary indicating total amount of training examples that fall into each node
    - max_depth: Maximum depth of the tree
    - max_prunings: Maximum amount of prunings
    - max_indivs: Maximum amount of individuals
    - max_generations: Maximum amount of generations
    - max_accuracy: Maximum accuracy
    - max_fpr: Maximum FPR
    """

    def __init__(self, struc, num_gen, num_indiv, prob_cross, prob_mutation):
        self.struc = struc
        self.num_gen = num_gen
        self.num_indiv = num_indiv
        self.prob_cross = prob_cross
        self.prob_mutation = prob_mutation
        self.population = None
        self.__initial_population()
    



    def __initial_population(self):
        """
        Creates the initial population. As it is defined, it may create some
        empty individuals (full tree)

        Parameters:
        - s_prob: Dictionary with the probability of each node for being selected
                in each individual. (Preferably generated by space_probs function)
        - num_indiv: Number of individuals to be generated

        Returns:
        - indivs: List of individuals forming the population
        """

        self.population = []
        space_probs = self.struc.space_probs()      # Probability of including each possible node in the new individual
        for i in range(self.num_indiv):          # For each individual to be created
            newset = []                     # Create its representation
            for a in space_probs:                # For each node in the tree
                r = np.random.rand()        # Calculate if that node will be selected (pruned)
                if r < space_probs[a]:           # If it is selected, append it to the representation
                    newset.append(a)
            
            self.population.append(Individual(self.struc, newset))    # Create and append the individual


    # TODO MEJORAR: Puede que haya cruces extremos en los que un árbol se queda virgen. Igualmente, puede que haya podas que se coman las unas a las otras.
    def __crossover(self, indiv1, indiv2):
        """
        Crossover between 2 individuals
        
        Parameters:
        - indiv1: First individual
        - indiv2: Second individual

        Returns:
        - newindiv1: First child individual
        - newindiv2: Second child individual
        """
        
        # If both individuals represent no prunings, they will be returned
        if len(indiv1.repr) == 0 and len(indiv2.repr) == 0:
            return indiv1, indiv2

        # In any other case
        # We will take the prunings done in each individual, join them, and select randomly to which individual we be assigned.
        newlist = []
        newlist.extend(indiv1.repr)
        newlist.extend(indiv2.repr)

        newindiv1 = []
        newindiv2 = []

        # Randomly assign the prunings to the individuals, with equal probability
        for elem in newlist:
            if np.random.rand() < 0.5:
                newindiv1.append(elem)
            else:
                newindiv2.append(elem)

        return Individual(self.struc, newindiv1), Individual(self.struc, newindiv2)


    # TODO torneo binario pero teniendo en cuenta únicamente una función de fitness univariable.
    def __tournament(self):
        """
        Applies tournament criterion over population, returning parents population.

        Parameters:
        - population: population of individuals to b

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

            # We select the node with the greates fitness value.
            fitness_1 = self.population[rand1].fitness
            fitness_2 = self.population[rand2].fitness
            print(fitness_1)
            print(fitness_2)
            print(fitness_1 > fitness_2)
            if fitness_1 > fitness_2:
                new_pop.append(self.population[rand1])
            else:
                new_pop.append(self.population[rand2])

        return new_pop


    def __pop_crossover(self):
        """
        Applies crossover over parents population

        Parameters:
        - population: population of parents
        - prob_cross: probability of crossover between parents

        Returns:
        - new_pop: new children population
        """
        
        new_pop = []                    # Children population

        longi = len(self.population)         
        for i in range(int(longi/2)):       # For each pair or parents
            rand = np.random.random()       # We decide if crossover will be done
            if rand < self.prob_cross:           # If so, we apply it
                new_indiv1, new_indiv2 = self.__crossover(self.population[2*i], self.population[(2*i)+1])
                new_pop.append(new_indiv1)
                new_pop.append(new_indiv2)
            else:                           # If not, we return the same parents
                new_pop.append(self.population[2*i])
                new_pop.append(self.population[(2*i)+1])

        return new_pop



    def __aux_random_value_prob_dict(self, prob_dict, rand):
        """
        Return an element given a probability dictionary, and a random value
        This is an auxiliary function

        Parameters:
        - prob_dict: dictionary with probabilities. They have to add up to 1
        - rand: random value between 0 and 1

        Returns:
        - k: selected key from the dictionary
        """
        total = 0
        for k, v in prob_dict.items():
            total += v
            if rand <= total:
                return k


    def __mutation(self, indiv):
        """
        Applies mutations randomly over an individual. The mutations may not happen at all

        Parameters:
        - indiv: individual for which the fitness will be calculated
        - space: space of prunings
        - base_leaves: dictionary with the leaves of the base tree.
        - prob_mutation: probability of mutation

        Returns:
        - fitness_val: its fitness value
        """

        rand_modify = np.random.random()

        if rand_modify < self.prob_mutation:       # If a mutation will be done
            
            # We will calculate the nodes from which a modification of a pruning can be done
            leaves = []

            # To begin with, we will find all the leaf nodes the pruned tree has
            for leaf in self.struc.base_leaves:
                leaves.append(leaf)
            
            # After that, we will check if there is any actual pruning of the tree that is set to be done that would
            # prune any of the already calculated leaves. If so, we will have to remove the previous leaves and add the new one

            for elem in indiv.repr:                                  # Prunings done (will be add and corrected)
                # We will calculate if it covers a leaf node. If so, we will erase it
                length = len(elem)
                leaves = [leaf for leaf in leaves if leaf[:length] != elem]
            
            # We will now directly add them to the current leaves.
            for elem in indiv.repr:
                leaves.append(elem)


            # We will now randomly select one of those leaves. We will select it using equal probabilities
            probs = {leaf: 1/len(leaves) for leaf in leaves}

            leaf = self.__aux_random_value_prob_dict(probs, np.random.random())         # Select one of the leaves

            # After having selected the leaf, we will now select the mutation applied over that leaf.

            new_repr = indiv.repr.copy()
            print("------")
            print(type(new_repr))

            if leaf in self.struc.base_leaves:                 # If the leaf is a leaf node of the complete tree, we can only select its parent
                if len(leaf) > 1:                       # It its parent is not the root node
                    new_repr.add(leaf[:-1])                    # We directly add it to the individual
            else:                                   # In other case we can go up or down in the tree hierarchy.
                new_probs = {}                          # We will create a new space for the possible mutations
                if len(leaf) > 1:
                    new_probs[leaf[:-1]] = 1/5          # Parent node
                children = self.struc.children_nodes(leaf)
                
                # Children nodes. We have to take into account the possibility of removing the pruning if a children is a leaf node of the actual tree
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

                prun = self.__aux_random_value_prob_dict(new_probs, np.random.random())         # Select one of them
                new_repr.remove(leaf)                                                      # Get rid of the leaf

                if prun != 'empty':                                                     # Add the pruning if necessary
                    new_repr.add(prun)

            return Individual(self.struc, new_repr)

        return(indiv)
        

    """
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
    """

    # TODO Completar con el esquema general del algoritmo genético
    def genetic_optimization(self, seed):
        """
        Defines the whole optimization process
        """

        np.random.seed(seed)

        print("comienzo")
        print(self.population)
        for i in range(self.num_gen):
            new_pop = self.__tournament()
            new_pop = self.__pop_crossover()
            new_pop = [self.__mutation(indiv) for indiv in new_pop]
            
            print(i)
            self.population = new_pop
        print("fin")
        for indiv in self.population:
            print(indiv.fitness)




# TESTS

struc = Tree_Structure(wc_data, prot, wc_target, clf)
gen_process = Genetic_Pruning_Process(struc, 2000, 50, 0.7, 0.2)
gen_process.genetic_optimization(777)

