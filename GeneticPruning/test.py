from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import numpy as np

# Leemos los datos de iris
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()



# Creamos una variable nueva que sea protegida (lo haremos en base a la primera variable)
prot = iris.data[:,0]
mean = prot.mean()
prot = np.where(prot < mean, 0, 1)


iris.data = np.c_[iris.data, prot]
# Mostramos los datos


clf.fit(iris.data, iris.target)
#print(clf.get_depth())
#print(clf.get_n_leaves())

#print(cross_val_score(clf, iris.data, iris.target, cv=10))

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("iris") 

print(clf.tree_)


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


def prunning_space(clf):
    """
    This function defines the prunning (search) space. 
    
    Parameters:
    - clf: Sklearn binary classification tree from which the prunings will be made

    Returns:
    - repr_space: Tuple containing all possible node codes from which the prunings
                  can be done
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    repr_space = []
    node_repr = np.zeros(shape=n_nodes).tolist()
    stack = [(0, [])]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, repr = stack.pop()
        node_repr[node_id] = repr

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


def fitness(indiv):
    """
    Calculates the fitness value of a certain individual. For doing so, the actual
    pruning is calculated over the base classifier and its results are used

    Parameters:
    - indiv: individual for which the fitness will be calculated

    Returns:
    - fitness_val: its fitness value
    """
    pass


def crossover(indiv1, indiv2):
    """
    Return correct individuals. The values conforming an individual may have inconsistencies
    due to prunings in some nodes generate the same trees as affect an already prunned branch
    
    Parameters:
    - indiv1: First individual
    - indiv2: Second individual

    Returns:
    - newindiv1: First child individual
    - newindiv2: Second child individual
    """
    pass




def mutation(prob_modify, prob_new_prun):
    """
    Applies a mutation

    Parameters:
    - indiv: individual for which the fitness will be calculated

    Returns:
    - fitness_val: its fitness value
    """
    pass




def genetic_optimization():
    pass



np.random.seed(0)

space = prunning_space(clf)
probs = space_probs(space)
print(initial_population(probs, 10))