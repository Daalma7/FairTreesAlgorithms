import numpy as np
import copy
import pandas as pd

from sklearn.tree._tree import TREE_LEAF

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
        self.fair_dict, self.total_samples_dict, self.base_leaves = self.aux_objectives_metrics()
        self.pruning_space = self.calc_pruning_space()

    def aux_objectives_metrics(self):
        """
        This function calculates different structures for the base classifier in order to be more
        efficient calculating each individual objectives metrics
        
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
    

    def calc_pruning_space(self):
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
    - objectives: objectives values for the individual
    """

    def __init__(self, struc, objs_string, repr):
        self.struc = struc
        self.objs_string = objs_string
        self.repr = self.struc.correct_indiv(repr)   # Each individual created will be automatically corrected in order to ensure consistency
        self.objectives = self.objectives()     # And also its objectives values will be calculated


    def objectives(self):
        """
        Calculates the objectives value of a certain individual. For doing so, the actual
        pruning is calculated over the base classifier and its results are used.

        Parameters:
        - indiv: individual for which the objectives will be calculated
        - base_leaves: Representation of the leaves of the tree
        - fair_dict: Fairness structure
        - total_samples_dict: Dictionary indicating total amount of training examples that fall into each node

        Returns:
        - objectives_val: its objectives value
        """
        
        # First of all, we will calculate the real leaf nodes of the individual, given
        # the prunings that will be applied.

        ret = []

        # First of all we will build the general confusion matrix divided by protected attribute.
            # First dimension: Actual class
            # Second dimension: Predicted class
        
        leaf_nodes = self.struc.base_leaves.copy()
        for elem in self.repr:
            leaf_nodes.add(elem)
        leaf_nodes = self.struc.correct_indiv(leaf_nodes)
        
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

        for obj in self.objs_string:

            if obj == 'accuracy':

                # We will now calculate the accuracy of this individual knowing the leaf nodes
                acc = []
                num_indivs = []
                for elem in leaf_nodes:
                    num = self.struc.number_of_indivs(elem)         
                    acc.append(self.struc.node_accuracy(elem) * num)    # Accuracy of the leaf node
                    num_indivs.append(num)                              # Number of individuals in the leaf node
                
                ret.append(np.sum(np.array(acc) / sum(num_indivs)))
                continue
            
            if obj == 'fpr_diff':
                fpr_0 = conf_mat_0[0][1] / (conf_mat_0[0][1] + conf_mat_0[0][0])
                fpr_1 = conf_mat_1[0][1] / (conf_mat_1[0][1] + conf_mat_1[0][0])
                
                ret.append(1-abs(fpr_0-fpr_1))

        return ret
    
    def dominates(self, other):
        """
        Checks if the individual dominates another one.

        Parameters:
        - other: other individual to be checked

        Returns:
        - True if the individual dominates the other one, False otherwise
        """
        ret = True
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                ret = False
                break
        return ret




class Individual_NSGA2(Individual):

    def __init__(self, struc, repr):
        Individual.__init__(struc, repr)
        self.rank = None
        self.crowding_distance = None

