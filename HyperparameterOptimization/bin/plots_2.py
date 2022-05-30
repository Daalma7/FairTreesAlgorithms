import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from math import ceil
import random
import csv
import warnings
from sklearn import datasets
import sys
import plotly
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import re
import os
import shutil

sys.path.append("..")
from general.ml import *
from general.problem import *

def wait_for_input():
    print("Press enter key to continue: ")
    input()

def plot_4d_interactive_plot(data, pre_objectives, title, filename):
    """
    Shows a 4d interactive plot being the last objective the color, and the rest represented in a 3d cartesian space
    """

    objectives = [objectives_dict.get(x) for x in pre_objectives]
    # Set marker properties
    markercolor = data[objectives[3]]

    if len(objectives) == 5:
        fig1 = px.scatter_3d(data, x=objectives[0], y=objectives[1], z=objectives[2],
                        color=objectives[3], opacity=0.7, color_continuous_scale='portland', title=title, symbol=objectives[4])
    if len(objectives) == 4:
        fig1 = px.scatter_3d(data, x=objectives[0], y=objectives[1], z=objectives[2],
                        color=objectives[3], opacity=0.7, color_continuous_scale='portland', title=title)
    
    fig1.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                          ticks="outside"))
    
    fig1.write_image(filename + ".png")
    fig1.write_html(filename + ".html")

def plot_compas(data, pre_objectives, title, filename):

    objectives = [objectives_dict.get(x) for x in pre_objectives]

    data_cpy = data.copy()
    data_cpy = data_cpy[objectives]
    
    color_discrete_map = {'nsga2': 'rgb(83,114,171)', 'smsemoa': 'rgb(209,136,92)', 'grea':'rgb(106,165,110)', 'Our method': 'rgb(65,105,225)', 'COMPAS': 'rgb(255,0,0)'}
    fig1 = px.scatter_3d(data_cpy[objectives], x=objectives[0], y=objectives[1], z=objectives[2],
                        color='algorithm', color_discrete_map=color_discrete_map, opacity=0.7, title=title)

    fig1.write_image(filename + ".png")
    fig1.write_html(filename + ".html")


#data is supposed to be 2d data
def plot_2d_mean_pareto(nonsortdata, pre_objectives, title, filename):

    objectives = [objectives_dict.get(x) for x in pre_objectives]

    data = nonsortdata.sort_values(by=objectives[0], axis=0, ascending=True)          #We order the values by the 1st component
    num_clus = min(int(np.rint(np.sqrt(data.shape[0]))), 30)                           #Number of clusters to do
    cut_points=[0]
    
    for i in range(1, num_clus+1):                                      #Cut points where we're going to split our data
        cut_points.append(int(np.rint(data.shape[0] * i / num_clus)))

    new_data = pd.DataFrame({objectives[0]:[], objectives[1]:[]})
    point_data = pd.DataFrame({objectives[0]:[], objectives[1]:[]})
    for i in range(num_clus):
        cut_data = data[cut_points[i]:cut_points[i+1]]
        new_data = pd.concat([new_data, cut_data])
        new_data.iloc[cut_points[i]:,0] = new_data.iloc[cut_points[i]:,0].mean()
        point_data = pd.concat( [point_data, pd.DataFrame({objectives[0]: [new_data.iloc[cut_points[i]:,0].mean()], objectives[1]: [new_data.iloc[cut_points[i]:,1].mean()]})] )
    
    sns.relplot(x=objectives[0], y=objectives[1],
            kind="line", ci="sd", data=new_data, color="red")
    sns.scatterplot(x=objectives[0], y=objectives[1], data=nonsortdata, size=20, alpha=0.6, zorder=0)
    sns.scatterplot(x=objectives[0], y=objectives[1], data=point_data, size=20, alpha=0.8, color="red", zorder=1)
    plt.xlabel(pre_objectives[0])
    plt.ylabel(pre_objectives[1])

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_3d_mean_pareto(data, pre_objectives, num_clus_per_side, title, filename):
    """
    Calculate and plot the mean 3d pareto individuals. For calculating those "mean individuals", for which there's no predefined way,
    so we're going to use a clustering approach. What we're going to do is calculate a "triangle" of initial points, using baricentric coordinates
    w.r.t. the worst point for each objective. After done, we're going to use k-means 
    """
    
    objectives = [objectives_dict.get(x) for x in pre_objectives]   #Objective calculation
    data_use = data[objectives].copy()                              #Define the data we're going to use

    initial_points = []                     #Points where clusters will be initially placed
    max_objectives = [data_use[x].max() for x in objectives]    #Maximum value for each objective
    
    for i in range(num_clus_per_side):
        for j in range(num_clus_per_side):
            print("a")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_scatter_matrix(data, pre_objectives, title, filename, size=20, alpha=0.7, palette=None):
    
    objectives = [objectives_dict.get(x) for x in pre_objectives]
    fig, ax = plt.subplots()
    sns.set_theme(style="ticks")
    """
    if len(pre_objectives) == 5:
        #g = sns.pairplot(data[objectives], hue=objectives[-1], plot_kws={"s": size, "alpha": alpha}, palette=palette)
        g = sns.pairplot(data[objectives], kind='reg', order=2, hue=objectives[-1])
    if len(pre_objectives) == 4:
        #g = sns.pairplot(data[objectives], plot_kws={"s": size, "alpha": alpha}, palette=palette)
        g = sns.pairplot(data[objectives], kind='reg', order=2)

    g.fig.suptitle(title, y=1)
    
    plt.savefig(filename, dpi=300)
    """
    if len(pre_objectives) == 5:
        if 'COMPAS' in pd.unique(data['algorithm']):
            palette['COMPAS'] = '#000000'    
        g = sns.PairGrid(data[objectives], hue=objectives[-1], palette=palette)
        
    else:
        g = sns.PairGrid(data[objectives])

    g = g.map_offdiag(sns.regplot, scatter=True, lowess=True, scatter_kws={'alpha':0.15})
    g = g.map_diag(sns.histplot, kde=True, multiple='layer')
    
    #g.fig.suptitle(title, y=1)
    g.add_legend(ncol=3)
    
    for lh in g.legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50]
    
    g.fig.get_children()[-1].set_bbox_to_anchor((0.5, 1, 0, 0))

    plt.savefig(filename, dpi=300, bbox_inches='tight')

def plot_correlation_heatmap(data, title, filename):
    
    f, ax = plt.subplots(figsize=(10, 6))
    corr = data.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_parallel_coordinates(data, pre_objectives, title, filename):
    
    objectives = [objectives_dict.get(x) for x in pre_objectives]

    data_2 = data[objectives].copy()

    data_2['algorithm'] = data_2['algorithm'].map({'nsga2': 1, 'smsemoa': 2, 'grea': 3})

    #data_2['algorithm'] = data_2['algorithm'].map({'nsga2': 1, 'smsemoa':2, 'grea': 3})
    fig = px.parallel_coordinates(data_2, dimensions=objectives[:-1],
                                color="algorithm", range_color=[0.5, 3.5],
                                color_continuous_scale=[(0.00, "darkorange"),   (0.33, "darkorange"),
                                                        (0.33, "darkcyan"), (0.66, "darkcyan"),
                                                        (0.66, "darkmagenta"),  (1.00, "darkmagenta")])

    fig.update_layout(coloraxis_colorbar=dict(
        title=title,
        tickvals=[1,2,3],
        ticktext=["nsga2","smsemoa","grea"],
        lenmode="pixels", len=100,
    ))
    
    fig.write_image(filename + ".png")
    fig.write_html(filename + ".html")



alg = dat = var = obj = mod = False        #Possible parameters given

#Dictionary to propperly create individuals given the objectives
objectives_dict = {'gmean_inv': 'error_val', 'dem_fpr': 'dem_fpr_val', 'dem_ppv': 'dem_ppv_val', 'dem_pnr': 'dem_pnr_val','num_leaves': 'num_leaves', 'data_weight_avg_depth': 'data_weight_avg_depth',
                   'algorithm': 'algorithm', 'criterion': 'criterion', 'creation_mode': 'creation_mode'}

message = "\nScript for getting cool and representative plots for the data"

message += "\nThe following parameters have been given by the user:"
error = False
for i in range(1, len(sys.argv)):           #We're going to read all parameters
    valid = False

    message += "\n- " + sys.argv[i]
    param = sys.argv[i].split('=')
    
    if not valid and param[0] == 'alg':               #Algorithms to use
        alg = valid = True
        algorithm = param[1].split(',')

    if not valid and param[0] == 'dat':               #Database name
        dat = valid = True
        dataset = param[1]
    
    if not valid and param[0] == 'var':               #Sensitive attribute name
        var = valid = True
        variable = param[1]
    
    if not valid and param[0] == 'model':
        model = valid = True
        model = param[1]
    
    if not valid and param[0] == 'obj':               #Objectives to take into account that we'll try to miminize
        obj = valid = True
        objectives = param[1].split(',')
        objdict = {'gmean_inv': gmean_inv, 'dem_fpr': dem_fpr, 'dem_ppv': dem_ppv, 'dem_pnr': dem_pnr, 'num_leaves': num_leaves, 'data_weight_avg_depth': data_weight_avg_depth}
        objectives = [objdict[x] for x in objectives]
    
    if not valid and param[0] == 'help':              #The user wants help
        print('\nThis file contains functions for calculating plots and graphs for better understanding the data structure, results and a better interpretation of them:\n\n\
\t- alg=(comm separated list of algorithms): Algorithms from which to take results. Possible algorithms are nsga2, smsemoa and grea. The default is nsga2,smsemoa,grea\n\n\
\t- dat=(dataset): Name of the dataset in csv format. The file should be placed at the folder named data. Initial dataset are adult, german, propublica_recidivism, propublica_violent_recidivism and ricci. The default is german.\n\n\
\t- var=(variable): Name of the sensitive variable for the dataset variable. Sensitive considered variables for each of the previous datasets are: adult-race, german-age, propublica_recidivism-race, propublica_violent_recidivism-race, ricci-Race. The default is the first variable of a the dataset (It is absolutely recommendable to change)\n\n\
\t- model=(model abbreviation): Model to use. Possible models are Decision Tree (DT) and Logistic Regression (LR). The default is DT.\n\n\
\t- obj=(comm separated list of objectives): List of objectives to be used. Possible objectives are: gmean_inv, dem_fpr, dem_ppv, dem_pnr, num_leaves, data_weight_avg_depth. You can add and combine them as you please. The default is gmean_inv,dem_fpr. IMPORTANT: Objectives should be written in the same order as they were written at the fairness.py execution sentence.\n\n\
\t- help: Shows this help and ends.\n\n\
An example sentence for execute this file could be:\n\n\
\tpython plots_2.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves\n\n\
Results are saved into the corresponding results/measures folder.')
        print("Execution succesful!\n------------------------------")
        sys.exit(0)
    
    if not valid:
        print('Some of the name of the parameters introduced is invalid.\n\
Please check out for mistakes there. Possible parameters are:\n\
alg, dat, var, obj\n')
        sys.exit(1)

print(message + "\n---")
    
#Now we're going to assign default values to each non intrudiced parameter
print("\nThe following parameters will be set by default:")
if not alg:
    algorithm = ['nsga2', 'smsemoa', 'grea']
    stralg = algorithm[0]
    for i in range(1, len(algorithm)):
        stralg += "," + algorithm[i]
    print("- alg=" + stralg)
if not dat:
    dataset = 'german'      #Default dataset: german
    print("- dat=" + dataset)
if not var:
    variable = pd.read_csv('../data/' + dataset + '.csv').columns[0]           #Default sensitive variable: First dataset variable
    print("- var=" + variable)
if not model:
    model = "DT"
    print("- model=" + model)
if not obj:
    objectives = [gmean_inv, dem_fpr] #Objectives to take into account that we'll try to miminize
    strobj = objectives[0].__name__
    for i in range(1, len(objectives)):
        strobj += "," + objectives[i].__name__
    print("- objectives=" + strobj)
print('---')



#String describing objectives
str_obj = objectives[0].__name__
for i in range(1, len(objectives)):
    str_obj = str_obj + '__' + str(objectives[i].__name__)


data = [pd.read_csv("../results/general_pareto_fronts/pareto_front_" + dataset + "_var_" + variable + "_model_" + model + "_obj_" + str_obj + ".csv")]
names = ["general"]

# Now we will find all files that contain pareto fronts due to the execution of an algorithm considering the parameters
regex = re.compile("general_individuals_pareto_" + dataset + ".*_var_" + variable + ".*_model_" + model + ".*_obj_" + str_obj + ".csv")

#File location
for x in algorithm:
    rootdir = "../results/" + x + "/individuals"
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                names.append(x)
                data.append(pd.read_csv(rootdir + "/" + file))


path = "../results/images/" + dataset + "_var_" + variable + '_model_' + model + "_obj_" + str_obj  #Create or override base directory
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

for i in range(len(data)):

    print("- Creating images for " + names[i])

    if dataset == "propublica_recidivism":
        ap = pd.DataFrame({'error_val': [0.34925611], 'dem_fpr_val': [0.213924], 'dem_ppv_val': [0.038379], 'num_leaves': [data[i]['num_leaves'].max()], 'algorithm': ['COMPAS']})

    if dataset == "propublica_violent_recidivism":
        ap = pd.DataFrame({'error_val': [0.38281533], 'dem_fpr_val': [0.196779], 'dem_ppv_val': [0.016405], 'num_leaves': [data[i]['num_leaves'].max()], 'algorithm': ['COMPAS']})

    if i > 0:
        data[i]['algorithm'] = names[i]
        objectives.append('algorithm')

    J=2
    if dataset == "propublica_recidivism" or dataset == "propublica_violent_recidivism":
        for j in range(J):
            data[i] = data[i].append(ap)
    
    path = "../results/images/" + dataset + "_var_" + variable + "_model_" + model + "_obj_" + str_obj + "/" + names[i] #Create or override specific directory
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if dataset == "propublica_recidivism" or dataset == "propublica_violent_recidivism":
        plot_compas(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'algorithm'], title="Comparing COMPAS to our solutions for " + dataset, filename=path + "/COMPAS_3d_plot")
        print("COMPAS_3d_plot Done")


    # Scatter matrices
    plot_scatter_matrix(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'algorithm'], title="Scatter matrix of the " + names[i] + " pareto front, with color representing algorithm", alpha=0.9, filename=path + "/Scatter_matrix_color_algorithm", palette={'nsga2': '#006fca', 'smsemoa': '#fc791e', 'grea': '#00ff00'})
    print("Scatter_matrix_color_algorithm Done")

    if dataset == "propublica_recidivism" or dataset == "propublica_violent_recidivism":
        data[i] = data[i].iloc[:-J,:]

    data[i]['criterion'] = data[i]['criterion'].map(lambda x: int(x))
    data[i]['criterion'] = data[i]['criterion'].map({0:'gini', 1:'entropy'})

    palette={"gini": 'gold', "entropy": 'purple'}
    plot_scatter_matrix(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'criterion'], title="Scatter matrix of the " + names[i] + " pareto front, with color representing criterion", palette=palette, filename=path + "/Scatter_matrix_color_criterion")
    print("Scatter_matrix_color_criterion Done")

    palette={"inicialization": 'k', "crossover": 'lime', "mutation": 'magenta'}
    if dataset == "propublica_recidivism" or dataset == "propublica_violent_revidivism":
        palette["COMPAS"] = 'blue'
    plot_scatter_matrix(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'creation_mode'], title="Scatter matrix of the " + names[i] + " pareto front, with color representing creation_mode", palette=palette, filename=path + "/Scatter_matrix_color_creation")
    print("Scatter_matrix_color_creation Done")

    # Other 4d interactive plot
    if i == 0:
        plot_4d_interactive_plot(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'algorithm'], title="Interactive 4d plot for " + names[i] + " pareto front, with num_leaves as color", filename=path + "/4d_plot_color_num_leaves")
    else:
        plot_4d_interactive_plot(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves'], title="Interactive 4d plot for " + names[i] + " pareto front, with num_leaves as color", filename=path + "/4d_plot_color_num_leaves")
    print("4d_plot_color_num_leaves Done")

    if i == 0:
        plot_4d_interactive_plot(data[i], pre_objectives=['num_leaves', 'dem_fpr', 'dem_ppv', 'gmean_inv', 'algorithm'], title="Interactive 4d plot for " + names[i] + " pareto front, with gmean_inv as color", filename=path + "/4d_plot_color_gmean_inv")
    else:
        plot_4d_interactive_plot(data[i], pre_objectives=['num_leaves', 'dem_fpr', 'dem_ppv', 'gmean_inv'], title="Interactive 4d plot for " + names[i] + " pareto front, with gmean_inv as color", filename=path + "/4d_plot_color_gmean_inv")
    print("4d_plot_color_gmean_inv Done")

    if i == 0:
        plot_parallel_coordinates(data[i], pre_objectives=['gmean_inv', 'dem_fpr', 'dem_ppv', 'num_leaves', 'algorithm'], title="Parallel coordinate plots for " + names[i], filename=path + "/Parallel_coordinates")
        print("Parallel_coordinates Done")

    #Correlation heatmap
    plot_correlation_heatmap(data[i], 'Correlation Heatmap for the ' + names[i] + 'pareto optimal solutions', filename=path + "/Correlation_heatmap")
    print("Correlation_heatmap Done")

    #2d mean pareto 
    plot_2d_mean_pareto(data[i], pre_objectives=['gmean_inv', 'dem_fpr'], title="2d mean pareto for " + names[i] + "pareto front", filename=path + "/Mean_2d_pareto")
    print("Mean_2d_pareto Done")


print("Execution succesful!")