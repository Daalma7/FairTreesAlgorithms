from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from CodigoTFM._classes import DecisionTreeClassifier
import numpy as np

# Leemos los datos de iris
clf = DecisionTreeClassifier(random_state=0, criterion="gini_fair", f_lambda=0.2)
iris = load_iris()

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
