# Development of fairness-aware algorithms based on Decision Trees || Master's thesis

![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)
![Cython](https://img.shields.io/badge/cython-yellow.svg?logo=data:image/png;base64,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)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-%23013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/seaborn-%236ba1af.svg?logo=seaborn&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?logo=github&logoColor=white)

![Status](https://img.shields.io/badge/status-finished-green)
![License](https://img.shields.io/badge/license-MIT-red)


This repository contains the work done on the implementation and testing of 3 
different multi-objective algorithms, which include methods for achieving a balance 
between classification and fairness. Puedes leer la memoria del proyecto en el archivo
[report.pdf](report.pdf).

## Brief descrition of developed algorithms
- **FairDT (FDT)**: Modification of the impurity criterion calculation during decision tree training to also consider fairness. The general form it has is:

$$(1-\lambda) * \text{gini/entropy} - \lambda * \text{fairness criterion} $$
- **Fair Genetic Pruning (FGP)**: Consideration of the matrix decision tree (largest decision tree which can be built using the available data that perfectly classifies the training set) for a task and pruning it based on objectives considered.
- **FairLGBM (FLGBM)**: Modification of the loss function in the LightGBM algorithm to incorporate fairness.


## Brief description of the experimentation
La experimentación ha consistido en probar cada algoritmo con 10 conjuntos de datos
muy reconocidos en el mundo de la justicia en el aprendizaje automático (adult, 
compas, diabetes, dutch, german, insurance, obesity, parkinson, ricci y student) 
utilizando 10 semillas aleatorias distintas para cada uno. Con los resultados obtenidos en cada
ejecución para cada algoritmo, se han calculado resultados medios. Los
algoritmos con los que se realizó experimentación fue con los 3 algoritmos desarrollados
además de con un árbol de decisión (DT).

Los hiperparámetros que definen el espacio de decisión de cada algoritmo son los siguientes:

- **DT**:
    - **criterion**: gini / entropy.
    - **max_depth**: profundidad máxima del árbol.
    - **min_samples_split**: mínima cantidad de individuos que deben caer sobre un nodo para poder dividirlo en 2 nodos hijos.
    - **max_leaf_nodes**: cantidad máxima de nodos hoja que puede tener el árbol final.
    - **class_weight**: peso que se da a cada una de las clases a predecir.
- **FDT**:
    - **mismos parámetros**, y adicionalmente:
    - **fair_param**: Parámetro que controla la proporción entre el criterio de impureza y el criterio de justicia durante el aprendizaje del árbol
- **FGP**
    - El propio método es ya en sí un algoritmo genético que devuelve una gran cantidad de soluciones. En lugar de optimización de hiperparámetros de un clasificador base, este método se aplica directamente.
- FLGBM
    - **num_leaves**: número de hojas que tendrá el árbol.
    - **min_data_in_leaf**: mínimo cantidad de datos que necesitará un nodo para poder dividirse.
    - **max_depth**: profundidad máxima del árbol.
    - **learning_rate**: tasa de aprendizaje del algoritmo.
    - **n_estimators**: número de clasificadores débiles a construir.
    - **feature_fraction**: proporción de características utilizadas para construir el modelo.
    - **fair_param**: controla la importancia entre la función de pérdida estándar (logloss) del algoritmo, con la función de justicia considerada.

Los objetivos a minimizar durante la experimentación han sido:

- **Inverted G-mean** (gmean_inv): El criterio de media geométrica se define como la raiz del producto de la tasa de verdaderos positivos y la de verdaderos negativos $$\sqrt{\text{TPR} \cdot \text{TNR}}$$. Al tratarse de un objetivo de minimización, se usará $$1-\sqrt{\text{TPR} \cdot \text{TNR}}$$.
- **Difference in False Positive Rate** (FPR$$_{\text{diff}$$): Resulta de la diferencia entre las probabilidades $$|P[p=1|Y=0,A=0]-P[p=1|Y=0,A=1]|$$, siendo $$p$$ el predictor utilizado, $$Y$$ el atributo a predecir, y $$A$$ un atributo sensible.


## Results

Los resultados han mostrado que se pueden encontrar soluciones mucho más justas y precisas utilizando los algoritmos empleados que utilizando un árbol de decisión normal.

TODO: Falta
## Libraries and dependencies:

You can try with higher versions, with all libraries but cython
- **python**=3.10.12
- **matplotlib**=3.8.3
- **pandas**=2.2.1
- **scikit-learn**=1.4.1.post1
- **pydotplus**=2.0.2
- **imblearn**
- **cython**=0.29.37
- **lightgbm**=4.3.0 (from the official lightgbm webpage)
- **seaborn**=0.13.2
- **pygmo**=2.19.5
<!-- conda create --name NAME conda-forge python=3.10.12 -->
<!-- conda activate NAME -->
<!-- pip install matplotlib -->
<!-- pip install pandas -->
<!-- pip install scikit-learn -->
<!-- pip install pydotplus -->
<!-- pip install imblearn -->
<!-- pip install cython=0.29.37 -->
<!-- execute build.sh inside /HyperparameterOptimization/models/FairDT -->
<!-- install lightgbm with cuda support from the lightgbm webpage -->
<!-- pip install seaborn -->
<!-- pip install pygmo -->

--- 

## Additional info
- Author: David Villar Martos
- Collaborators: David Villar Martos
- Director del proyecto: Jorge Casillas Barranquero
