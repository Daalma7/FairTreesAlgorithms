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

# TODO incorporar medidas de fairness
def aux_fitness_metrics(clf, prot, y):
    """
    This function calculates different structures for the base classifier in order to be more
    efficient calculating each individual fitness metrics
    
    Parameters:
    - clf: Sklearn binary classification tree from which the prunings will be made
    - prot: Protected attribute
    - y

    Returns:
    - fair_dict: Dictionary indicating well classified examples inside that node
    - total_samples_dict: Dictionary indicating total amount of training examples that fall in
                          that node
    - base_leaves: Leave nodes codes that the base tree has 
    """
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    values = clf.tree_.value

    total_samples_dict = {}
    base_leaves = []
    fair_dict = {}

    assoc_dict = {}         # Association dictionary to relate node_ids with our current
                            # representation

    stack = [(0, [])]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, repr = stack.pop()
        assoc_dict[node_id] = repr

        total_samples_dict[tuple(repr)] = np.sum(values[node_id])

        template = []
        for i in range(pd.unique(y).shape[0]):
            template.append([])
            for j in range(pd.unique(prot).shape[0]):
                template[i].append(0)
        fair_dict[tuple(repr)] = template

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], repr + [0]))
            stack.append((children_right[node_id], repr + [1]))
        else:
            base_leaves.append(tuple(repr))
    
    # We will now create the fairness structure
        # First dimension for specifying the class
        # Second dimension for specifying the protected attribute
    
    # We will update each node in the fair structure with the information of the
    # training data
    node_indicator = clf.decision_path(wc_data)
    leaf_id = clf.apply(wc_data)
    for sample_id in range(wc_data.shape[0]):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                fair_dict[tuple(assoc_dict[node_id])][y[sample_id]][prot[sample_id]] += 1
                continue
            fair_dict[tuple(assoc_dict[node_id])][y[sample_id]][prot[sample_id]] += 1

    

    return fair_dict, total_samples_dict, set(base_leaves)


def node_accuracy(repr, fair_dict):

    acc = []
    for elem in fair_dict[repr]:
        acc.append(sum(elem))

    return max(acc) / number_of_indivs(repr, fair_dict)


def number_of_indivs(repr, fair_dict):

    num = 0
    for elem in fair_dict[repr]:
        for elem2 in elem:
            num += elem2
    
    return num


def fitness(indiv, base_leaves, fair_dict, total_samples_dict,):
    """
    Calculates the fitness value of a certain individual. For doing so, the actual
    pruning is calculated over the base classifier and its results are used.

    Parameters:
    - indiv: individual for which the fitness will be calculated

    Returns:
    - fitness_val: its fitness value
    """
    
    # First of all, we will calculate the real leaf nodes of the individual, given
    # the prunings that will be applied.

    leaf_nodes = base_leaves.copy()
    for elem in indiv:
        leaf_nodes.add(elem)
    leaf_nodes = correct_indiv(leaf_nodes)

    # We will now calculate the accuracy of this individual knowing the leaf nodes

    acc = []
    num_indivs = []
    for elem in leaf_nodes:
        num = number_of_indivs(elem, fair_dict)
        acc.append(node_accuracy(elem, fair_dict) * num)
        num_indivs.append(num)
    
    print(np.sum(np.array(acc) / sum(num_indivs)))
    return np.sum(np.array(acc) / sum(num_indivs))



def prunning_space(clf):
    """
    This function defines the prunning (search) space. 
    
    Parameters:
    - clf: Sklearn binary classification tree from which the prunings will be made

    Returns:
    - repr_space: Tuple containing all possible node codes from which the prunings
                  can be done
    """
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
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


def space_probs(space):
    """
    Defines the probability of each node for being selected in each initial population
    individual. It is calculated inversely proportional to each node depth, and depends
    to the actual depth of the tree. The maximium probability is initialy fixed to 0.5

    Parameters:
    - space: Tuple containing all nodes where prunings might be done

    Returns:
    - dict_prob: Dictionary containing the probability for each node of being selected
    """
    maxlen = 0
    dict_prob = {}
    for a in space:
        if len(a) > maxlen:
            maxlen = len(a)
        dict_prob[a] = len(a)
    
    dict_prob = {a: 1/(2**(maxlen-v+1)) for a, v in dict_prob.items()}
    return dict_prob



def initial_population(s_prob, num_indiv):
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
    indivs = []
    for i in range(num_indiv):
        newset = []
        for a in s_prob:
            r = np.random.rand()
            if r < s_prob[a]:
                newset.append(a)
        indivs.append(correct_indiv(set(newset)))
    
    return indivs

            
def correct_indiv(indiv):
    """
    Return correct individuals. The values conforming an individual may have inconsistencies
    due to prunings in some nodes generate the same trees as affect an already prunned branch
    
    Parameters:
    - indiv: Set representing an individual to which correct its representation

    Returns:
    - indiv: Corrected individual
    """
    
    if len(indiv) > 0:
        indiv = list(indiv)
        new_indiv = []
        add_list = np.zeros(len(indiv)).tolist()    # Binary addition list

        # Calculation of hierarchical covered prunings
        for i in range(len(indiv)):
            for j in range(i+1, len(indiv)):
                if indiv[i] == indiv[j][:len(indiv[i])]:
                    add_list[j] = 1
        
        # Cleaning of those prunings
        for i in range(len(indiv)):
            if add_list[i] < 1:
                new_indiv.append(indiv[i])
        return set(new_indiv)
    else:
        return indiv

def apply_pruning(indiv):
    """
    Apply a given pruning to the base clasifier

    Parameters:
    - indiv: individual representing the pruning

    Returns:
    - new_clf: prunned clasifier
    """
    new_clf = copy.deepcopy(clf)

    # For each pruning to do:
    for prun in indiv:
        # We will convert the pruning code into the node in which the pruning
        # will be done
        cur_node_index = 0
        for i in range(len(prun)):
            if prun[i] == 0:
                cur_node_index = new_clf.tree_.children_left[cur_node_index]
            else:
                cur_node_index = new_clf.tree_.children_right[cur_node_index]
        
        # We remove the children of that node
        new_clf.tree_.children_left[cur_node_index] = TREE_LEAF
        new_clf.tree_.children_right[cur_node_index] = TREE_LEAF
        
    return new_clf

def crossover(indiv1, indiv2):
    """
    Crossover between 2 individuals
    
    Parameters:
    - indiv1: First individual
    - indiv2: Second individual

    Returns:
    - newindiv1: First child individual
    - newindiv2: Second child individual
    """
    
    # If both individuals represent no prunnings, they will be returned
    if len(indiv1) == 0 and len(indiv2) == 0:
        return indiv1, indiv2
    
    # If only 1 individual has some prunings, we will translate a random
    # pruning to the one without them
    if len(indiv1) == 0:
        newindiv1 = list(indiv1.copy())
        newindiv2 = list(indiv2.copy())
        change = np.random.randint(0, len(newindiv2))
        newindiv1.append(newindiv2[change])
        del newindiv2[change]

        return set(newindiv1), set(newindiv2)
    
    if len(indiv2) == 0:
        newindiv1 = list(indiv1.copy())
        newindiv2 = list(indiv2.copy())
        change = np.random.randint(0, len(newindiv1))
        newindiv2.append(newindiv1[change])
        del newindiv1[change]

        return set(newindiv1), set(newindiv2)

    # In other cases, both individuals has at least 1 pruning
    # We will identify in this case common prunning branches and swap them
    # This will enhace local exploration
    common_branches = []

    for a in indiv1:
        for b in indiv2:
            if a[:min(len(a), len(b))] == b[:min(len(a), len(b))]:
                common_branches.append([a,b])

    if len(common_branches) == 0:   # If there are no common branches
        # In this case we will swap two random prunings
        newindiv1 = list(indiv1.copy())
        newindiv2 = list(indiv2.copy())
        change1 = np.random.randint(0, len(newindiv1))
        change2 = np.random.randint(0, len(newindiv2))
        newindiv1.append(newindiv2[change2])
        newindiv2.append(newindiv1[change1])
        del newindiv1[change1]
        del newindiv2[change2]

        return correct_indiv(set(newindiv1)), correct_indiv(set(newindiv2))
    else:
        # We will randomly select one of them and change those prunnings
        change = np.random.randint(0, len(common_branches))
        newindiv1 = indiv1.copy()
        newindiv2 = indiv2.copy()
        newindiv1.remove(common_branches[change][0])
        newindiv2.remove(common_branches[change][1])
        newindiv1.add(common_branches[change][1])
        newindiv2.add(common_branches[change][0])

        return correct_indiv(newindiv1), correct_indiv(newindiv2)    

def children_nodes(pruning, space):
    """
    Returns which are the children nodes from a pruning space, if they exist.

    Parameters:
    - pruning: the pruning code from which calculate the children nodes
    - space: the total children space

    Returns:
    - children: the children codes of the pruning, if any.
    """

    children = []
    pos_0 = pruning + (0,)
    pos_1 = pruning + (1,)
    if pos_0 in space:
        children.append(pos_0)
    if pos_1 in space:
        children.append(pos_1)
    
    return children

def tournament(population, base_leaves, well_dict, total_samples_dict):
    """
    Applies tournament criterion over population.

    Parameters:
    - population: population of individuals to b

    Returns:
    - new_pop: parents population
    """
    
    new_pop = []

    longi = len(population)
    for i in range(longi):
        rand1 = np.random.randint(0, longi)
        rand2 = np.random.randint(0, longi)
        while rand1 == rand2:
            rand2 = np.random.randint(0, longi)

        if fitness(population[rand1], base_leaves, well_dict, total_samples_dict) > fitness(population[rand2], base_leaves, well_dict, total_samples_dict):
            new_pop.append(population[rand1])
        else:
            new_pop.append(population[rand2])

    return new_pop

def pop_crossover(population, prob_cross):
    """
    Applies crossover over parents population

    Parameters:
    - population: population of parents
    - prob_cross: probability of crossover between parents

    Returns:
    - new_pop: new children population
    """
    
    new_pop = []

    longi = len(population)
    for i in range(int(longi/2)):
        rand = np.random.random()
        if rand < prob_cross:
            new_indiv1, new_indiv2 = crossover(population[2*i], population[(2*i)+1])
            new_pop.append(new_indiv1)
            new_pop.append(new_indiv2)
        else:
            new_pop.append(population[2*i])
            new_pop.append(population[(2*i)+1])

    return new_pop



def aux_random_value_prob_dict(prob_dict, rand):
    """
    Return an element given a probability dictionary, and a random value
    This is an auxiliary function

    Parameters:
    - prob_dict: dictionary with probabilities
    - rand: random value between 0 and 1

    Returns:
    - k: selected key from the dictionary
    """
    total = 0
    for k, v in prob_dict.items():
        total += v
        if rand <= total:
            return k

def mutation(indiv, space, base_leaves, prob_modify, prob_new_prun):
    """
    Applies mutations randomly over an individual. The mutations may not happen at all

    Parameters:
    - indiv: individual for which the fitness will be calculated
    - space: space of prunings
    - prob_modify: probability of a pruning to be modified
    - prob_new_prun: probability of a pruning to be created

    Returns:
    - fitness_val: its fitness value
    """

    rand_modify = np.random.random()
    rand_new_prun = np.random.random()

    if rand_modify < prob_modify and len(indiv) > 0: # If a pruning modification will be done
        # We first select a random pruning:
        prun = list(indiv)[np.random.randint(len(indiv))]

        # We first calculate the possible new prunings to be considered with respect
        # to the selected one
        possible_pruns = {}

        new_space = space.copy()

        for leaf in base_leaves:
            new_space.append(leaf)

        childs = children_nodes(prun, new_space)
        for child in childs:
            possible_pruns[child] = 2
        if len(prun) > 1:
            possible_pruns[prun[:-1]] = 1
        
        sum_val = float(sum(possible_pruns.values()))
        
        possible_pruns = {key: value/sum_val for key, value in possible_pruns.items()}

        if len(possible_pruns) > 0:
            rand_val = np.random.random()
            selected = aux_random_value_prob_dict(possible_pruns, rand_val)
            indiv.remove(prun)
            if(not selected in base_leaves):
                indiv.add(selected)

        
    if rand_new_prun < prob_new_prun: # If a new prunning is going to be produced
        new_space = prunning_space(apply_pruning(indiv)) #
        new_prob_space = space_probs(new_space)
        sum_val = float(sum(new_prob_space.values()))
        new_prob_space = {key: value/sum_val for key, value in new_prob_space.items()}


        if len(new_prob_space) > 0:
            rand_val = np.random.random()
            selected = aux_random_value_prob_dict(new_prob_space, rand_val)
            indiv.add(selected)
    
    return correct_indiv(indiv)


# TODO
def genetic_optimization(clf, prot, y, seed, num_indiv, num_gen, p_cross, p_mod, p_new):
    """
    Defines the whole optimization process
    """

    np.random.seed(seed)

    well_dict, total_samples_dict, base_leaves = aux_fitness_metrics(clf, prot, y)
    space = prunning_space(clf)
    probs = space_probs(space)
    pop = initial_population(probs, num_indiv)

    print("comienzo")
    print(pop)
    for i in range(num_gen):
        pop = tournament(pop, base_leaves, well_dict, total_samples_dict)

        pop = pop_crossover(pop, p_cross)

        for indiv in pop:
            indiv = mutation(indiv, space, base_leaves, p_mod, p_new)
    print("fin")
    print(pop)




# TESTS
genetic_optimization(clf, prot, wc_target, 0, 50, 2000, 0.7, 0.2, 0.05)

