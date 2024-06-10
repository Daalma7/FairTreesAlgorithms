import numpy as np
import copy
import pandas as pd

from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score


class Tree_Structure:

    """
    Tree structure which is used to control the Genetic Pruning processes.
    It contains structures in order to the process to be as time efficent as possible
    """

    def __init__(self, x_train, y_train, prot_train, x_val=None, y_val=None, prot_val=None, seed=None, combine_train_val=None):

        """
        Class constructor
            Parameters:
                - x_train: Prediction information with which our base decision tree will be trained
                - y_train: Class information with which our base decision tree will be trained
                - prot_train: Protected attribute information with which our base decision tree will be trained
                - x_val: Prediction information values for validation
                - y_val: Class information values for validation
                - prot_val: Protected attribute valuess for validation
                - seed: Random seed
                - combine_train_val: Combines training and validation samples
        """

        self.x_train = x_train
        self.y_train = y_train
        self.prot_train = prot_train
        
        self.x_val = x_val
        self.y_val = y_val
        self.prot_val = prot_val

        if x_val is None:
            self.x_val = x_train
        
        if y_val is None:
            self.y_val = y_train
        
        if prot_val is None:
            self.prot_val = prot_train
        
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        if combine_train_val is True:
            new_x = pd.concat([x_train, x_val], ignore_index=True)
            new_y = pd.concat([y_train, y_val], ignore_index=True)
            new_prot = pd.concat([prot_train, prot_val], ignore_index=True)
            self.x_train = new_x
            self.x_val = new_x
            self.y_train = new_y
            self.y_val = new_y
            self.prot_train = new_prot
            self.prot_val = new_prot
                    
        np.random.seed(self.seed)
        self.clf = DecisionTreeClassifier(random_state=self.seed)
        self.clf.fit(x_train, y_train)

        self.fair_dict, self.total_samples_dict, self.base_leaves, self.assoc_dict, self.val_nodes = self.aux_objectives_metrics()
        self.pruning_space = self.calc_pruning_space()

        

    def aux_objectives_metrics(self, verbose=False):
        """
        This function calculates different structures for the base classifier in order to be more
        efficient calculating each individual objectives metrics
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
        
        # We get the decision path of all our validation samples
        # First component is the id
        # Second component is the node ids where that sample goes through
        paths = self.clf.decision_path(self.x_train).nonzero()        
        
        fair_template = []                               # We create a list of lists for the fair dict dictionary
        for i in range(pd.unique(self.y_train).shape[0]):      # First dimension represents the class
            fair_template.append([])
            for j in range(pd.unique(self.prot_train).shape[0]):   # Second dimension represents the protected attribute
                fair_template[i].append(0)
        
        list_ids = [] # List of node ids
        stack = [(0, [])]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, repr = stack.pop()         # Extract the first node id and its representation
            list_ids.append(node_id)
            
            assoc_dict[node_id] = repr          # Create the representation for that node.

            # If the left and right child of a node is not the same, we are dealing with a split node (non leaf)
            is_split_node = children_left[node_id] != children_right[node_id]
        
            if is_split_node:                # If a split node, append left and right children and depth to `stack` so we can loop through them
                stack.append((children_left[node_id], repr + [0]))  # Append tree considered and our path based representations.
                stack.append((children_right[node_id], repr + [1]))
            else:                           # If a leaf node, append the node id to `base_leaves`
                base_leaves.append(repr)  # Append our representation to the list of leaves to return
                
        # We will now create the fairness structure
            # First dimension for specifying the class
            # Second dimension for specifying the protected attribute    
        for elem in list_ids:                               # We initialize it for each node_id
            fair_dict[elem] = copy.deepcopy(fair_template)
                
                
        for i in range(len(paths[0])):         #for every individual and node from which goes through
            fair_dict[paths[1][i]][self.y_train.iloc[paths[0][i]]][self.prot_train.iloc[paths[0][i]]] += 1        # We build our fair_dict
                    
        #Lastly, we change the faif_dict keys to the current used ids.
        fair_dict = {tuple(assoc_dict[k]): v for k, v in fair_dict.items()}
        total_samples_dict = {key: np.sum(fair_dict[key]) for key in fair_dict}   # Total amount of training samples in that node

        val_nodes = []
        
        # We will update each node in the fair structure with the information of the validation data
        node_indicator = self.clf.decision_path(self.x_val)             # First of all we obtain the decision path for each training example
        #leaf_id = self.clf.apply(self.x_val)                            # And we compute where each specific instance falls.
        
        for sample_id in range(self.x_val.shape[0]):               # For each training example
            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]

            # We update the fair structure
            for node_id in node_index:
                check = assoc_dict[node_id]
                if not check in val_nodes:
                    val_nodes.append(check)
                
        if verbose:
            print("fair_dict", fair_dict)
            print("total_samples_dict: ", total_samples_dict)
            print("base_leaves: ", base_leaves)
        
        print(len(val_nodes), len(total_samples_dict), "Diferencia:", len(total_samples_dict) - len(val_nodes), ', Relativa:', len(val_nodes) / len(total_samples_dict))
        
        return fair_dict, total_samples_dict, base_leaves, assoc_dict, val_nodes
    
    
        """
        n_nodes = self.struc.clf.tree_.node_count
        children_left = self.struc.clf.tree_.children_left
        children_right = self.struc.clf.tree_.children_right
        feature = self.struc.clf.tree_.feature
        threshold = self.struc.clf.tree_.threshold
        values = self.struc.clf.tree_.value
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
        
            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        
        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node with value={value}.".format(
                        space=node_depth[i] * "\t", node=i, value=values[i]
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node with value={value}: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                        value=values[i],
                    )
                )
    """
    
    def calc_pruning_space(self, verbose=False):
        """
        This function defines the pruning (search) space. 
            Returns:
                - repr_space: Tuple containing all possible node codes from which the prunings can be done
        """

        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        repr_space = []
        stack = [(0, [])]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, repre = stack.pop()

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                repr_space.append(repre)
                stack.append((children_left[node_id], repre + [0]))
                stack.append((children_right[node_id], repre + [1]))
                
        if verbose:
            print("repr_space", repr_space[1:])
        
        return repr_space
    
    
    # MODIFIABLE
    def space_probs(self):
        """
        Defines the probability of each node for being selected if our population
        in each initial population individual. 
        The strategy which is followed is to give each depth the same probability to be prunned.
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



    def node_accuracy(self, repre):
        """
        Calculates the accuracy of a given node of the tree.
            Parameters:
                - repr: Representation of the node
            Returns:
                - accuracy: Accuracy of the node
        """

        acc = []
        for elem in self.fair_dict[tuple(repre)]:
            acc.append(sum(elem))
            
        return max(acc) / self.number_of_indivs(repre)



    def number_of_indivs(self, repre):
        """
        Calculates the number of individuals in a given node.
            Parameters:
                - repr: Representation of the node
            Returns:
                - num: Amount of individuals in the node
        """

        num = 0
        for elem in self.fair_dict[tuple(repre)]:
            for elem2 in elem:
                num += elem2
        
        return num

    
    def children_nodes(self, sing_prun):
        """
        Returns which are the children nodes from a pruning space, if they exist.
            Parameters:
                - sing_prun: the pruning code from which calculate the children nodes
            Returns:
                - children: the children codes of the pruning, if any. They are returned as tuples
        """

        children = []

        pos_0 = sing_prun.copy()          # Codes of the possible children nodes
        pos_1 = sing_prun.copy()
        pos_0.append(0)
        pos_1.append(1)

        if pos_0 in self.pruning_space:              # If the children node exists in the space of possible prunings, we return them
            children.append(tuple(pos_0))
        if pos_1 in self.pruning_space:
            children.append(tuple(pos_1))
        
        return children
    

    def node_id_to_repre(self, ids):
        """
        Returns individual's representation using the nodes id of the prunings, instead
        of the code which represent how to travel the tree to get to them
            Returns:
                - id_repre: individual's representation in terms of node ids
        """
        
        new_repre = []

        for elem in ids:
            new_repre.append(self.assoc_dict[elem])

        return new_repre



class Individual:
    """
    Class that represents an individual. An individual in the optimization process consist of a series
    of prunings over the base classifier tree.
    """
    
    def __init__(self, struc, objs_string, repre, creation_mode, objectives=None, objectives_train=None):

        """
        Class constructor
            Parameters:
                - struc: tree structure with all calculated metrics, from the general.Tree_Structure class
                - objs_string: Strings defining the objective functions for the optimization process
                - repr: Representation of the individual
        """

        self.struc = struc
        self.objs_string = objs_string
        self.repre = repre   # Each individual created needs to have a minimal representation
        if objectives is None or objectives_train is None:
            self.objectives, self.objectives_train = self.calc_objectives()     # And also its objectives values will be calculated
        else:
            self.objectives = objectives
            self.objectives_train = objectives_train
        self.creation_mode = creation_mode
        self.num_prunings = len(repre)
        self.depth = None
        self.num_leaves = None
        self.depth_unbalance = None
        self.data_avg_depth = None

    # MODIFIABLE
    def calc_objectives(self):
        """
        Calculates the objectives value of a certain individual. For doing so, the actual
        pruning is calculated over the base classifier and its results are used.
        Result metrics for the optimization process are computed used validation data.
            Returns:
                - objectives_val: its objectives value
        """

        true_clf = self.get_tree()

        ret_val = None
        ret_train = None

        # For that purpose, we will build the general confusion matrix divided by protected attribute.
            # conf_mat_x: protected attribute value
            # First dimension: Actual class
            # Second dimension: Predicted class
        
        """
        leaf_nodes = self.struc.base_leaves.copy()

        for elem in self.repr:
            leaf_nodes.append(elem)
        leaf_nodes = self.struc.correct_indiv(leaf_nodes)
        """

        for elem in ['val', 'train']:
            ret = []
            x = None
            y_real = None
            prot = None

            if elem == 'val':
                x = self.struc.x_val
                y_real = self.struc.y_val
                prot = self.struc.prot_val
            elif elem == 'train':
                x = self.struc.x_train
                y_real = self.struc.y_train
                prot = self.struc.prot_train
            
            y_pred = true_clf.predict(x)
            pred_df = pd.DataFrame({'y_val': y_real, 'y_pred': y_pred, 'prot': prot})
            
            pred_df_p = pred_df.loc[pred_df['prot'] == 1]        #p variables represent data belonging to privileged class
            pred_df_u = pred_df.loc[pred_df['prot'] != 1]        #u variables represent data belonging to unprivileged class
            y_val_p = pred_df_p['y_val']
            y_val_u = pred_df_u['y_val']
            y_pred_p = pred_df_p['y_pred']
            y_pred_u = pred_df_u['y_pred']
            
            tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()    # Confusion matrix divided by each protected attribute value
            tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
            
            #print(tn_p, fp_p, fn_p, tp_p, tn_u, fp_u, fn_u, tp_u)        

            for obj in self.objs_string:
                """
                You can implement whichever objective you want 
                """

                if obj == 'accuracy':

                    ret.append(1-accuracy_score(y_real, y_pred))
                    continue
                
                if obj == 'tpr_diff':
                    tpr_p = tp_p / (tp_p + fn_p)
                    tpr_u = tp_u / (tp_u + fn_u)
                    ret.append(abs(tpr_p-tpr_u))
                    continue
                
                if obj == 'fpr_diff':
                    fpr_p = fp_p / (fp_p + tn_p)
                    fpr_u = fp_u / (fp_u + tn_u)
                    ret.append(abs(fpr_p-fpr_u))
                    continue
                    
                if obj == 'ppv_diff':
                    ppv_p = tp_p / (tp_p + fp_p)
                    ppv_u = tp_u / (tp_u + fp_u)
                    ret.append(abs(ppv_p-ppv_u))
                    continue

                if obj == 'pnr_diff':
                    pnr_p = fn_p + tn_p / (fn_p + tn_p + fp_p + tp_p)
                    pnr_u = fn_u + tn_u / (fn_u + tn_u + fp_u + tp_u)
                    ret.append(abs(pnr_p - pnr_u))
                    continue
                
                if obj == 'gmean_inv':
                    ret.append(1-geometric_mean_score(y_real, y_pred))

            if elem == 'val':
                ret_val = ret
            elif elem == 'train':
                ret_train = ret
        return ret_val, ret_train
    
    
    def dominates(self, other):
        """
        Checks if the individual dominates another one.
        We understand this problem as a minimization one.
            Parameters:
                - other: other individual to be checked
            Returns:
                - True if the individual dominates the other one, False otherwise
        """
        and_condition = True
        or_condition = False
        self_obj = [0.9 * self.objectives[i] + 0.1 * self.objectives_train[i] for i, _ in enumerate(self.objectives)]
        other_obj = [0.9 * other.objectives[i] + 0.1 * other.objectives_train[i] for i, _ in enumerate(other.objectives)]
        #for first, second in zip(self.objectives, other.objectives):
        for first, second in zip(self_obj, other_obj):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        
        """
        if not(or_condition) and and_condition:          # If both individuals have the same objective values:
            for first, second in zip(self.objectives_train, other.objectives_train):
                and_condition = and_condition and first <= second
                or_condition = or_condition or first < second
        """
        return (and_condition and or_condition)


    
    def get_tree(self):
        """
        Returns the actual tree, given the current representation
            Returns:
                - The DecisionTreeClassifier associated with the current individual's representation
        """
    
        # We will modify in this case the internal tree structure. More precisely, the structure controlling right and left children
        new_tree = copy.deepcopy(self.struc.clf)
        children_left = new_tree.tree_.children_left
        children_right = new_tree.tree_.children_right
        values = new_tree.tree_.value
        all_depths = []
        all_samples = []

        #values = new_tree.tree_.value
        
        stack = [(0, [])]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, repr = stack.pop()         # Extract the first node id and its representation

            # We now check if the current node is in our pruning list.
            pruning = False
            if repr in self.repre:                  # In case we have to prune it, we flag it
                pruning = True

                
            # If the left and right child of a node is not the same, we are dealing with a split node (non leaf)
            is_split_node = children_left[node_id] != children_right[node_id]
        
            if is_split_node and not pruning:                # If a split node, and no prunins,
                                                             # append left and right children and depth to `stack` so we can loop through them
                stack.append((children_left[node_id], repr + [0]))  # Append tree considered and our path based representations.
                stack.append((children_right[node_id], repr + [1]))
            else:                           # If a leaf node, or a pruning to be applied, we overwrite the values of children to be -1 (no children)
                new_tree.tree_.children_left[node_id] = TREE_LEAF
                new_tree.tree_.children_right[node_id] = TREE_LEAF
                all_depths.append(len(repr))
                all_samples.append(np.sum(values[node_id]))

        all_depths = np.array(all_depths)
        all_samples = np.array(all_samples)

        #new_tree.tree_.n_leaves = len(all_depths)
        #new_tree.tree_.max_depth = all_depths.max()
        self.depth = all_depths.max()
        self.num_leaves = len(all_depths)
        self.depth_unbalance = float(all_depths.min()) / float(all_depths.max())
        self.data_avg_depth = np.dot(all_depths, all_samples) / float(np.sum(all_samples)) # Mean depth weighted by samples
        return new_tree



    def test_tree(self, x_test, y_test, prot_test):
        """
        Calculates the objectives value of a certain individual. For doing so, the actual
        pruning is calculated over the base classifier and its results are used.
        Result metrics for the optimization process are computed used test data.
            Parameters:
                - x_test: x test information the same way as in x_train and x_val
                - y_test: y test information the same way as in y_train and y_val
                - prot_test: prot test information the same way as in prot_train and prot_val
            Returns:
                - objectives_val: its objectives value
        """

        true_clf = self.get_tree()

        ret = []

        # For that purpose, we will build the general confusion matrix divided by protected attribute.
            # conf_mat_x: protected attribute value
            # First dimension: Actual class
            # Second dimension: Predicted class
        
        """
        leaf_nodes = self.struc.base_leaves.copy()

        for elem in self.repr:
            leaf_nodes.append(elem)
        leaf_nodes = self.struc.correct_indiv(leaf_nodes)
        """
        
        y_pred = true_clf.predict(x_test)
        pred_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'prot': prot_test})
        
        pred_df_p = pred_df.loc[pred_df['prot'] == 1]        #p variables represent data belonging to privileged class
        pred_df_u = pred_df.loc[pred_df['prot'] != 1]        #u variables represent data belonging to unprivileged class
        y_test_p = pred_df_p['y_test']
        y_test_u = pred_df_u['y_test']
        y_pred_p = pred_df_p['y_pred']
        y_pred_u = pred_df_u['y_pred']
        
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test_p, y_pred_p).ravel()    # Confusion matrix divided by each protected attribute value
        tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_test_u, y_pred_u).ravel()
        
        #print(tn_p, fp_p, fn_p, tp_p, tn_u, fp_u, fn_u, tp_u)        

        for obj in self.objs_string:
            """
            You can implement whichever objective you want 
            """

            if obj == 'accuracy':

                ret.append(1-accuracy_score(y_test, y_pred))
                continue
            
            if obj == 'tpr_diff':
                tpr_p = tp_p / (tp_p + fn_p)
                tpr_u = tp_u / (tp_u + fn_u)
                ret.append(abs(tpr_p-tpr_u))
                continue
            
            if obj == 'fpr_diff':
                fpr_p = fp_p / (fp_p + tn_p)
                fpr_u = fp_u / (fp_u + tn_u)
                ret.append(abs(fpr_p-fpr_u))
                continue
                
            if obj == 'ppv_diff':
                ppv_p = tp_p / (tp_p + fp_p)
                ppv_u = tp_u / (tp_u + fp_u)
                ret.append(abs(ppv_p-ppv_u))
                continue

            if obj == 'pnr_diff':
                pnr_p = fn_p + tn_p / (fn_p + tn_p + fp_p + tp_p)
                pnr_u = fn_u + tn_u / (fn_u + tn_u + fp_u + tp_u)
                ret.append(abs(pnr_p - pnr_u))
                continue
            
            if obj == 'gmean_inv':
                ret.append(1-geometric_mean_score(y_test, y_pred))
                pass

        return ret
    
    
    def repre_to_node_id(self):
        """
        Returns individual's representation using the nodes id of the prunings, instead
        of the code which represent how to travel the tree to get to them
            Returns:
                - id_repre: individual's representation in terms of node ids
        """
        
        id_repre = []
        
        key_list = list(self.struc.assoc_dict.keys())
        val_list = list(self.struc.assoc_dict.values())

        for elem in self.repre:
            if type(elem) is list:
                position = val_list.index(elem)
            else:
                position = val_list.index(list(elem))


            id_repre.append(key_list[position])
        return id_repre
    



class Individual_NSGA2(Individual):
    """
    Class representing an individual for a NSGA2 optimization process, within this context
    """

    def __init__(self, struc, objs_string, repre, creation_mode, objectives=None, objectives_train=None):
        """
        Class constructor
            Parameters:
                - struc: tree structure with all calculated metrics, from the general.Tree_Structure class
                - repr: Representation of the individual
        """
        Individual.__init__(self, struc, objs_string, repre, creation_mode, objectives, objectives_train)
        self.domination_count=0
        self.dominated_solutions = []
        self.rank = None
        self.crowding_distance = None