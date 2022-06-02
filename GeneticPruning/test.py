from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import pandas as pd
import graphviz


from genetic import Genetic_Pruning_Process_NSGA2
from general import Tree_Structure


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

# TESTS

struc = Tree_Structure(wc_data, prot, wc_target, clf)
gen_process = Genetic_Pruning_Process_NSGA2(struc, ['accuracy'], 2000, 50, 0.7, 0.2)
indivs = gen_process.genetic_optimization(777)
for indiv in indivs:
    print(indiv.repr)

