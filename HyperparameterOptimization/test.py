from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join('.', 'models')))
print(sys.path)

from FairDT._classes import DecisionTreeClassifier




# Para estas pruebas voy a trabajar con el conjunto de datos de wisconsin porque es binario y tiene
# mejor pinta que iris para este contexto de clasificación binaria con atributo protegido binario

wc_data = pd.read_csv("./models/wdbc.data", header=None)
# Generamos las etiquetas
wc_target = wc_data.iloc[:,1].replace(["B","M"], [0,1])
# Generamos los predictores
wc_data = wc_data.iloc[:,2:]
# Vamos a binarizar el primer atributo y lo vamos a considerar como protegido
prot = wc_data.iloc[:,0]
mean = prot.mean()
print(mean)
wc_data.iloc[:,0] = np.where(prot < mean, 0, 1)
prot = wc_data.iloc[:,0]
print(wc_data.shape)
# Mostramos los datos
print("Predictores de wisconsin:\n", wc_data)
print("Etiquetas de wisconsin:\n", wc_target)
print("Atributo protegido:\n", prot)

print("a")


clf = DecisionTreeClassifier(random_state=0, criterion="entropy_fair", f_lambda=0.1, fair_fun='ppv_diff')
print("antes del fit")
clf.fit(wc_data.to_numpy(), wc_target.to_numpy(), prot=prot.to_numpy())
print("después del fit")

print("Profundidad: ", clf.get_depth())
print("Número de hojas: ", clf.get_n_leaves())



"""
# Leemos los datos de iris
clf = DecisionTreeClassifier(random_state=0, criterion="gini_fair", f_lambda=0.2)
iris = load_iris()

# Lo convertimos en un problema binario
iris.data = iris.data[:100,:]
iris.target = iris.target[:100]


# Creamos una variable nueva que sea protegida (lo haremos en base a la primera variable)
prot = iris.data[:,0]
mean = prot.mean()
print(mean)
prot = np.where(prot < mean, 0, 1)
print(iris.data.shape)


iris.data = np.c_[iris.data, prot]
print(iris.data.shape)
# Mostramos los datos
print("Predictores de iris:\n", iris.data)
print("Etiquetas de iris:\n", iris.target)
print("Atributo protegido:\n", prot)


print("antes del fit")
clf.fit(iris.data, iris.target, prot=prot)
print("después del fit")
print(clf.get_depth())
print(clf.get_n_leaves())



#print(cross_val_score(clf, iris.data, iris.target, cv=10))

"""

