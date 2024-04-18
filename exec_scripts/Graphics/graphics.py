import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

datasetlist = ['academic','adult','arrhythmia','bank','catalunya','compas','credit','crime','default','diabetes-w','diabetes','drugs','dutch','german','heart','hrs','insurance','kdd-census','lsat','nursery','obesity', 'older-adults','oulad','parkinson','ricci','singles','student','tic','wine','synthetic-athlete','synthetic-disease','toy']
dict_outcomes = {'academic': 'atd','adult': 'income','arrhythmia': 'arrhythmia','bank': 'Subscribed','catalunya': 'recid','compas': 'score','credit': 'NoDefault','crime': 'ViolentCrimesPerPop','default': 'default','diabetes-w': 'Outcome','diabetes': 'readmitted','drugs': 'Coke','dutch': 'status','german': 'Label','heart': 'class','hrs': 'score','insurance': 'charges','kdd-census': 'Label','lsat':'ugpa','nursery': 'class','obesity': 'NObeyesdad','older-adults': 'mistakes','oulad': 'Grade','parkinson': 'total_UPDRS','ricci': 'Combine','singles': 'income','student': 'G3','tic': 'income', 'wine': 'quality','synthetic-athlete': 'Label','synthetic-disease': 'Label','toy': 'Label'}
dict_protected = {'academic': 'ge','adult': 'Race','arrhythmia': 'sex','bank': 'AgeGroup','catalunya': 'foreigner','compas': 'race','credit': 'sex','crime': 'race','default': 'SEX','diabetes-w': 'Age','diabetes': 'Sex','drugs': 'Gender','dutch': 'Sex','german': 'Sex','heart': 'Sex','hrs': 'gender','insurance': 'sex','kdd-census': 'Sex','lsat':'race','nursery': 'finance','obesity': 'Gender','older-adults': 'sex','oulad': 'Sex','parkinson': 'sex','ricci': 'Race','singles': 'sex','student': 'sex','tic': 'religion','wine': 'color','synthetic-athlete': 'Sex','synthetic-disease': 'Age','toy': 'sst'}

PATH_TO_RESULTS = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))) + '/results'

datalist = ['adult', 'compas', 'diabetes', 'dutch', 'german', 'insurance', 'obesity', 'parkinson', 'ricci']
models = ['FDT', 'GP']
ngen = 300
indiv = 150
bseed = 100
objectives = ['gmean_inv', 'dem_fpr']

obj_str ='__'.join(objectives)
print(obj_str)



def line_evolution(dataname, model, bseed, ngen, indiv, obj_str, metrics, include_ext=True, store=False):
    """
    Plot line plots showing evolution of different parameters through generations
        Parameters:
            - dataname: Name of the dataset
            - model: String containing the algorithm employed 
            - bseed: Base random seed
            - ngen: Number of generations
            - indiv: Number of individuals within the population
            - obj_str: String containing the objective functions
            - metrics: Metrics to take into account
            - include_ext: Include extra objectives or not 
            - store: Boolean attribute. If true, the graphic is stored. In any other case, it will be plotted on screen
    """
    if not model is 'GP':
        data = pd.read_csv(f"{PATH_TO_RESULTS}/{model}/nsga2/generation_stats/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{indiv}_model_{model}_obj_{obj_str}.csv")
    else:
        data = pd.read_csv(f"{PATH_TO_RESULTS}/{model}/generation_stats/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{indiv}_model_{model}_obj_{obj_str}.csv")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 3*len(metrics)), sharey=False)
    fig.suptitle(f"Metrics of Fair Decision Trees in each generation")
    plt.gcf().subplots_adjust(bottom=0.1)

    metric_colors = {'mean_leaves': 'forestgreen', 'std_leaves': 'lightgreen', 'ext_leaves': 'darkgreen',
                     'mean_depth': 'red', 'std_depth': 'lightsalmon', 'ext_depth': 'darkred',
                     'mean_data_avg_depth': 'dodgerblue', 'std_data_avg_depth':'lightblue' , 'ext_data_avg_depth': 'darkblue'}
    print("b")
    for i, metric in enumerate(metrics):
        data[f"mean_{metric}+std"] = data[f"mean_{metric}"] + data[f"std_{metric}"]
        data[f"mean_{metric}-std"] = data[f"mean_{metric}"] - data[f"std_{metric}"]
        if include_ext:
            sns.lineplot(ax = axes[i], x=range(data.shape[0]), y=data[f"max_{metric}"], label='max,min', color=metric_colors[f"ext_{metric}"])
            sns.lineplot(ax = axes[i], x=range(data.shape[0]), y=data[f"min_{metric}"], color=metric_colors[f"ext_{metric}"])
        sns.lineplot(ax = axes[i], x=range(data.shape[0]), y=data[f"mean_{metric}"], label = 'mean', color = metric_colors[f"mean_{metric}"])

        axes[i].fill_between(x=range(data.shape[0]), y1=data[f"mean_{metric}+std"], y2=data[f"mean_{metric}-std"], color=metric_colors[f"std_{metric}"])

        #axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Generation')
        axes[i].set_ylabel(f'{metric}')
    
    fig.tight_layout()

    if store:
        plt.savefig(f"{PATH_TO_RESULTS}/FDT/nsga2/graphics/{dataname}/{dataname}_evolution_lines.pdf", format="pdf")
    else:
        plt.show()




def plot_2d_solutions(dataname, model, bseed, ngen, indiv, obj_str, store=False):
    """
    Does a scatterplot of solutions
        Parameters:
            - dataname: Name of the dataset
            - model: String containing the algorithm employed
            - bseed: Base random seed
            - ngen: Number of generations
            - indiv: Number of individuals within the population
            - obj_str: String containing the objective functions
            - store: Boolean attribute. If true, the graphic is stored. In any other case, it will be plotted on screen
    """
    if not model is 'GP':
        data = pd.read_csv(f"{PATH_TO_RESULTS}/{model}/nsga2/generation_stats/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{indiv}_model_{model}_obj_{obj_str}.csv")
    else:
        data = pd.read_csv(f"{PATH_TO_RESULTS}/{model}/generation_stats/{dataname}/{dataname}_seed_{bseed}_var_{dict_protected[dataname]}_gen_{ngen}_indiv_{indiv}_model_{model}_obj_{obj_str}.csv")

    objectives = obj_str.split('__')

    data = data.sort_values(by=objectives[0], axis=0, ascending=True)          #We order the values by the 1st component
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
    sns.scatterplot(x=objectives[0], y=objectives[1], data=data, size=20, alpha=0.6, zorder=0)
    sns.scatterplot(x=objectives[0], y=objectives[1], data=point_data, size=20, alpha=0.8, color="red", zorder=1)
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])

    plt.title('Título')
    plt.tight_layout()
    if store:
        plt.savefig(f"{PATH_TO_RESULTS}/FDT/nsga2/graphics/{dataname}/{dataname}_pareto_front.png")
    else:
        plt.show()

def plot_scatter_matrix():
    pass

def plot_correlation_heatmap():
    pass




























"""
#data is supposed to be 2d data
def plot_2d_mean_pareto(nonsortdata, pre_objectives, title, filename):



def plot_3d_mean_pareto(data, pre_objectives, num_clus_per_side, title, filename):
"""
"""
    Calculate and plot the mean 3d pareto individuals. For calculating those "mean individuals", for which there's no predefined way,
    so we're going to use a clustering approach. What we're going to do is calculate a "triangle" of initial points, using baricentric coordinates
    w.r.t. the worst point for each objective. After done, we're going to use k-means 
"""
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

"""
for dataname in datalist:
    for model in models:
        line_evolution(dataname, model, bseed, ngen, indiv, obj_str, ['leaves', 'depth', 'data_avg_depth'], True, True)