import math
import pandas as pd
import numpy as np
import pygmo as pg

from general.individual import *
from collections import OrderedDict as od


#Here, there are implementations for calculating quality measures for the solutions generated by MOEAs.


objectives_val_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val', 'num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth'}    
objectives_tst_dict = {'gmean_inv': 'error_tst', 'dem_fpr': 'dem_fpr_tst', 'dem_ppv': 'dem_ppv_tst', 'dem_pnr': 'dem_pnr_tst', 'num_leaves': 'num_leaves_tst', 'data_weight_avg_depth': 'data_weight_avg_depth_tst'}    

#Hypervolume calculation of the set of solutions, using ref as the worst possible point.
#Measure to maximize
def hypervolume(list, ref):
    newlist = [x.objectives for x in list]          #List of objectives
    hv = pg.hypervolume(newlist)
    return hv.compute(ref, hv_algo = pg.hvwfg())

#Spacing, used to measure uniform spacing within population's solutions. If spacing is 0 it means that every individual's distance in ||.||_1
#to its nearest neighbour solution is the same as the distance in ||.||_1 to its nearest neighbour of any other individual
#Measure to maximize
def spacing(lis):
    distances = []          #Calculates minimal ||.||_1 distance of each individual to any other individual in the population
    for indiv in lis:
        dist = float('inf')
        for other_indiv in lis:
            cur_dist = 0
            ran = len(other_indiv.objectives)
            for i in range(ran):
                cur_dist += abs(indiv.objectives[i]-other_indiv.objectives[i])
            if cur_dist < dist and not cur_dist == 0 :
                dist = cur_dist
        distances.append(dist)
    mean_dist = np.mean(np.array(distances))
    spac = 0                #Final calculation
    for dist in distances:
        spac += (mean_dist-dist)**2
    return math.sqrt(spac / float(len(distances) -1))

#Maximum_spread: ||.||_2 of the vector which in coordinate i has the maximum differences between any two individuals in their ith objective
#Measures the spread of the population
#Measure to maximize
def maximum_spread(lis):
    ms = 0
    ran = len(lis[0].objectives)
    maxdists = []
    for i in range(ran):
        maxdists.append(0)
    for indiv in lis:
        for other_indiv in lis:
            for i in range(ran):
                cur_dist = (indiv.objectives[i]-other_indiv.objectives[i])**2
                if cur_dist > maxdists[i]:
                    maxdists[i] = cur_dist
    for i in range(ran):
        ms += maxdists[i]
    return math.sqrt(ms)


#Defines the proportion of individuals that belong to the actual pareto front.
#We cannot use id since there could be 2 individuals with different id that represent the same individual
#We use an epsilon = 0.0000001 constant for avoiding rounding errors or impressicions, but not considering an individual not belonging to the pareto front as so.
def error_ratio(list, pareto_optimal):
    total_in = 0
    for x in list:
        found = False
        i = 0
        while not found and i < len(pareto_optimal):
            if np.linalg.norm(np.array(x.objectives) - np.array(pareto_optimal[i].objectives)) < 0.0000001:
                total_in += 1
                found = True
            i+=1
    return 1 - (float(total_in) / len(list))

    
def overall_pareto_front_spread(list, pareto_optimal):
    
    #We first compute ideal an nadir points of the pareto optimal reference set:
    ip = ideal_point(pareto_optimal)
    np = nadir_point(pareto_optimal)

    iplist = ideal_point(list)
    nplist = nadir_point(list)

    prod = 1

    for i in range(len(ip)):
        if nplist[i] > np[i]:       #If we have a point which value on objective i is worse than nadir point's value
            prod *= ((np[i] - iplist[i]) / (np[i] - ip[i]))
        else:
            prod *= ((nplist[i] - iplist[i]) / (np[i] - ip[i]))

    return prod

def generational_distance(list, pareto_optimal):
    total_dist = 0
    for x in list:
        dist = float('inf')
        for y in pareto_optimal:
            new_dist = np.linalg.norm(np.array(x.objectives) - np.array(y.objectives))
            if new_dist < dist:
                dist = new_dist
        total_dist += dist

    return np.sqrt(total_dist) / len(list)

def inverted_generational_distance(list, pareto_optimal):
    total_dist = 0
    for x in pareto_optimal:
        dist = float('inf')
        for y in list:
            new_dist = np.linalg.norm(np.array(x.objectives) - np.array(y.objectives))
            if new_dist < dist:
                dist = new_dist
        total_dist += dist

    return np.sqrt(total_dist) / len(pareto_optimal)

def ideal_point(list):
    ideal = np.ones(len(list[0].objectives)).tolist()   #List with as much ones as objectives are.
    numobj = len(list[0].objectives)
    for x in list:                                      #We give it the least value for each coordinate of all found values
        for i in range(numobj):
            if x.objectives[i] < ideal[i]:
                ideal[i] = x.objectives[i]
    return ideal


def nadir_point(list):
    nadir = np.zeros(len(list[0].objectives)).tolist()   #List with as much zeros as objectives are.
    numobj = len(list[0].objectives)
    for x in list:                                      #We give it the least value for each coordinate of all found values
        for i in range(numobj):
            if x.objectives[i] > nadir[i]:
                nadir[i] = x.objectives[i]
    return nadir

def algorithm_proportion(df):
    return "\n" + str(df['algorithm'].value_counts(normalize=True)).split('Name')[0][:-1]


#If mean is > 0, val results are generally better than test ones, which is expected. If >> 0, there's overfit, if < 0, there's underfit
def diff_val_test_rate(df, obj):
    diff = []
    new_df = df[[objectives_val_dict.get(obj), objectives_tst_dict.get(obj)]]
    for i in range(df.shape[0]):
        diff.append(float(new_df.iloc[i,0]) - float(new_df.iloc[i,1]))      #val evaluation - test evaluation
    diff = np.array(diff)
    return np.mean(diff), np.std(diff)


def coverage(list1, list2):
    numcovered = 0
    for indiv in list2:                      #For each individual on the second list of individuals
        covered = False
        for other_indiv in list1:            #Let's see if an individual on the first set dominates it
            if other_indiv.dominates(indiv):
                covered = True
        if covered:
            numcovered += 1
    return float(numcovered) / len(list2)