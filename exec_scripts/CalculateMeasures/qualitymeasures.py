import math
import pandas as pd
import numpy as np
import pygmo as pg

from collections import OrderedDict as od
from collections import Counter


#Here, there are implementations for calculating quality measures for the solutions generated by MOEAs.


objectives_val_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val', 'num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth'}    
objectives_tst_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst', 'num_leaves': 'num_leaves_tst', 'data_weight_avg_depth': 'data_weight_avg_depth_tst'}    


def hypervolume(indivs):
    """
    Computes the hypervolume of the individuals
    Measure to maximize
        Parameters:
            - indivs: List containing pareto optimal individuals
        Returns:
            - Hypervolume of the individuals, considering the worst point as [1,1,...,1]
    """
    newlist = [x.objectives_test for x in indivs]          #List of objectives
    worst = [1 for obj in newlist[0]]
    for i in range(len(newlist)):
        if newlist[i] == worst:
            newlist[i] = [1-1e-10 for obj in newlist[0]]
    hv = pg.hypervolume(newlist)
    return hv.compute(worst, hv_algo = pg.hvwfg())





def spacing(indivs):
    """
    Spacing, used to measure uniform spacing within population's solutions. If spacing is 0 it means that every individual's distance in ||.||_1
    to its nearest neighbour solution is the same as the distance in ||.||_1 to its nearest neighbour of any other individual
    Measure to maximize
        Parameters:
            - indivs: List containing pareto optimal individuals
        Returns:
            - Spacing metric of these individuals
    """
    if len(indivs) < 2:
        return 0
    else:
        distances = []          #Calculates minimal ||.||_1 distance of each individual to any other individual in the population
        for indiv in indivs:
            dist = float('inf')
            for other_indiv in indivs:
                cur_dist = 0
                ran = len(other_indiv.objectives_test)
                for i in range(ran):
                    cur_dist += abs(indiv.objectives_test[i]-other_indiv.objectives_test[i])
                if cur_dist < dist and not cur_dist == 0 :
                    dist = cur_dist
            if dist == float('inf'):
                distances.append(0)
            else:
                distances.append(dist)
        mean_dist = np.mean(np.array(distances))
        spac = 0                #Final calculation
        for dist in distances:
            spac += (mean_dist-dist)**2
        return math.sqrt(spac / float(len(distances) -1))






def maximum_spread(indivs):
    """
    Returns maximum spread metric
    It is the ||.||_2 of the vector which in coordinate i has the maximum differences between any two individuals in their ith objective
    Measure to maximize
        Parameters:
            - indivs: List containing pareto optimal individuals
        Returns:
            - Maximum spread metric of these individuals
    """
    ms = 0
    ran = len(indivs[0].objectives_test)
    maxdists = []
    for i in range(ran):
        maxdists.append(0)
    for indiv in indivs:
        for other_indiv in indivs:
            for i in range(ran):
                cur_dist = (indiv.objectives_test[i]-other_indiv.objectives_test[i])**2
                if cur_dist > maxdists[i]:
                    maxdists[i] = cur_dist
    for i in range(ran):
        ms += maxdists[i]
    return math.sqrt(ms)



def error_ratio(list, pareto_optimal, round_error = 1e-7):
    """
    Defines the proportion of individuals that belong to the actual pareto front.
    We cannot use id since there could be 2 individuals with different id that represent the same individual
    We use an rounding error for avoding impressicions, but not considering an individual not belonging to the pareto front as so.
    Measure to minimize
        Parameters:
            - list: list of individuals to calculate error_ratio
            - pareto_optimal: Pareto optimal individuals to which compare
            - round_error: rounding error parameter, so that the distance between individuals should be less than it
        Returns:
            - Error ratio metric of these individuals
    """
    total_in = 0
    for x in list:
        found = False
        i = 0
        while not found and i < len(pareto_optimal):
            if np.linalg.norm(np.array(x.objectives_test) - np.array(pareto_optimal[i].objectives_test)) < round_error:
                total_in += 1
                found = True
            i+=1
    return 1 - (float(total_in) / len(list))

    



def overall_pareto_front_spread(list, pareto_optimal):
    """
    Defines overall pareto front metric. This metric compare the spread of a given solution list to that of the pareto front
    Measure to maximize
        Parameters:
            - list: list of individuals to calculate overall Pareto front spread
            - pareto_optimal: Pareto optimal individuals to which compare
        Returns:
            - Overall Pareto front spread metric of these individuals
    """
    # We first compute ideal an nadir points of the pareto optimal reference set:
    ip = ideal_point(pareto_optimal)
    np = nadir_point(pareto_optimal)

    iplist = ideal_point(list)
    nplist = nadir_point(list)

    prod = 1

    if not ip == np:
        for i in range(len(ip)):
            if nplist[i] > np[i]:       #If we have a point which value on objective i is worse than nadir point's value
                prod *= ((np[i] - iplist[i]) / (np[i] - ip[i]))
            else:
                prod *= ((nplist[i] - iplist[i]) / (np[i] - ip[i]))

    return prod





def generational_distance(list, pareto_optimal):
    """
    Defines overall pareto front metric. This metric compare the spread of a given solution list to that of the pareto front
    Measure to maximize
        Parameters:
            - list: list of individuals to calculate overall Pareto front spread
            - pareto_optimal: Pareto optimal individuals to which compare
        Returns:
            - Overall Pareto front spread metric of these individuals
    """
    total_dist = 0
    for x in list:
        dist = float('inf')
        for y in pareto_optimal:
            new_dist = np.linalg.norm(np.array(x.objectives_test) - np.array(y.objectives_test))
            if new_dist < dist:
                dist = new_dist
        total_dist += dist

    return np.sqrt(total_dist) / len(list)





def inverted_generational_distance(list, pareto_optimal):
    """
    Defines overall pareto front metric. This metric compare the spread of a given solution list to that of the pareto front
    Measure to maximize
        Parameters:
            - list: list of individuals to calculate overall Pareto front spread
            - pareto_optimal: Pareto optimal individuals to which compare
        Returns:
            - Overall Pareto front spread metric of these individuals
    """
    total_dist = 0
    for x in pareto_optimal:
        dist = float('inf')
        for y in list:
            new_dist = np.linalg.norm(np.array(x.objectives_test) - np.array(y.objectives_test))
            if new_dist < dist:
                dist = new_dist
        total_dist += dist

    return np.sqrt(total_dist) / len(pareto_optimal)





def ideal_point(list):
    """
    Calculates the ideal point using a solution set
        Parameters:
            - list: list of individuals to calculate ideal point
        Returns:
            - Ideal point for those individuals
    """
    ideal = np.ones(len(list[0].objectives_test)).tolist()   #List with as much ones as objectives_test are.
    numobj = len(list[0].objectives_test)
    for x in list:                                      #We give it the least value for each coordinate of all found values
        for i in range(numobj):
            if x.objectives_test[i] < ideal[i]:
                ideal[i] = x.objectives_test[i]
    return ideal





def nadir_point(list):
    """
    Calculates the nadir point using a solution set
        Parameters:
            - list: list of individuals to calculate nadir point
        Returns:
            - Nadir point for those individuals
    """
    nadir = np.zeros(len(list[0].objectives_test)).tolist()   #List with as much zeros as objectives_test are.
    numobj = len(list[0].objectives_test)
    for x in list:                                      #We give it the least value for each coordinate of all found values
        for i in range(numobj):
            if x.objectives_test[i] > nadir[i]:
                nadir[i] = x.objectives_test[i]
    return nadir





def algorithm_proportion(indivs):
    """
    Returns the proportion of individuals in a given individual list by each algorithm
        Parameters:
            - indivs: List of all individuals to calculate proportion
        Returns
            - dictionary containing each algorithm name as key, and its proportion as value
    """
    algorithms = Counter([indiv.algorithm for indiv in indivs])
    suma = sum(algorithms.values())
    #return {key: value/float(suma) for key, value in algorithms.items()}
    return {key: value for key, value in algorithms.items()}





def diff_val_test_rate(model, indivs, obj):
    """
    Calculates a comparison between validation and test results
    If mean is > 0, val results are generally better than test ones, which is expected.
    If >> 0, there's overfit, if < 0, there's underfit
        Parameters:
            - df: Dataframe
            - obj: Objectives 
        Returns
            - Mean and std of the differences between validation and test sets
    """
    return_df = {'algorithm':[], 'measure':[], 'val-test':[]}
    for i in range(len(obj)):
        for indiv in indivs:
            return_df['algorithm'].append(model)
            return_df['measure'].append(obj[i])
            return_df['val-test'].append(indiv.objectives[i] - indiv.objectives_test[i])
    
    return pd.DataFrame(return_df)






def coverage(indiv_cover, indiv_covered):
    """
    Calculates the coverage between two individual lists. It calculates the coverage of individuals from the first
    list to the second list. This is an assymetrical metric, so also the other configuration should be tested
        Parameters:
            - indiv_cover: List of individuals which are tested to dominate (cover) the ones from the other list
            - indiv_covered: List of individuals which are tested to be dominated (be covered) by the ones from the other list 
        Returns
            - Proportion of individuals covered from the first list to the second
    """
    numcovered = 0
    for indiv in indiv_covered:                      #For each individual on the second list of individuals
        covered = False
        for other_indiv in indiv_cover:            #Let's see if an individual on the first set dominates it
            if other_indiv.dominates(indiv):
                covered = True
        if covered:
            numcovered += 1
    return float(numcovered) / len(indiv_covered)