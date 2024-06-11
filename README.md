# Development of fairness-aware algorithms based on Decision Trees || Master's thesis

![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)
![Cython](https://img.shields.io/badge/cython-yellow.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1MTIiIGhlaWdodD0iNTEyIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiIgaW1hZ2UtcmVuZGVyaW5nPSJvcHRpbWl6ZVF1YWxpdHkiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZmlsbD0iI2VhZWJlYSIgeG1sbnM6dj0iaHR0cHM6Ly92ZWN0YS5pby9uYW5vIj48cGF0aCBkPSJNNTExLjUgMTQ3LjV2M2MtMzUuMzM1LjE2Ny03MC42NjggMC0xMDYtLjUtMjQuMjEzLTQ2LjEzMS02Mi4yMTMtNzMuMTMxLTExNC04MS00MC45MzEtNS41MTEtNzkuNTk4IDEuNDg5LTExNiAyMS0zMi40NDQgMjEuMTQtNTMuMjc4IDUwLjY0LTYyLjUgODguNS04LjgyMyA0Mi4xMjktOS44MjMgODQuNDYzLTMgMTI3IDcuMDM2IDQzLjkxMyAyNi41MzYgODEuMDc5IDU4LjUgMTExLjUgMjguMDc2IDIxLjk2OSA2MC4wNzYgMzIuNDY5IDk2IDMxLjUgMzguMTU3IDEuMzg3IDcyLjQ5MS05LjExMyAxMDMtMzEuNSAxMy4xNzMtMTEuMjA0IDIzLjg0LTI0LjM3IDMyLTM5LjVhMTcwMC41NiAxNzAwLjU2IDAgMCAxIDEwMS0uNWMtNDIuNTEzIDY3Ljk2NS0xMDMuODQ2IDEwNy4yOTgtMTg0IDExOC01NC4wMDYgNy41MTEtMTA3LjAwNiAzLjE3OC0xNTktMTNDODYuMTc3IDQ1NS4zNDQgMzcuNjc3IDQwNi4xNzcgMTIgMzM0LjVjLTYuMDUxLTIwLjA5LTEwLjIxNy00MC40MjMtMTIuNS02MXYtNDhjNS4wNzUtNTEuODEzIDI1LjU3NS05Ni40NzkgNjEuNS0xMzRDMTA4LjM3MyA0NC42OCAxNjUuNTQgMTguODQ3IDIzMi41IDE0YzU0LjQ4Mi01LjM2IDEwNy4xNDkgMS45NzQgMTU4IDIyIDUzLjMyNCAyMy4xNiA5My42NTcgNjAuMzI3IDEyMSAxMTEuNXoiIG9wYWNpdHk9Ii45ODMiLz48cGF0aCBkPSJNMjQ5LjUgMTAxLjVjMTQuMDA0LS4xNjcgMjguMDA0IDAgNDIgLjUgMTMuMjEyLjE3NCAyNS41NDUgMy41MDcgMzcgMTAgNy43NTYgNS42NjYgMTIuOTIyIDEzLjE2NiAxNS41IDIyLjUuNjY3IDI5IC42NjcgNTggMCA4Ny0zLjY2OCAxMy42NjctMTIuMTY4IDIyLjgzNC0yNS41IDI3LjVhMjE1MC4xOCAyMTUwLjE4IDAgMCAxLTkwIDJjLTIwLjIwMyAzLjg3MS0zMy4wMzcgMTUuNzA0LTM4LjUgMzUuNWEzNjQuMjcgMzY0LjI3IDAgMCAwLTEuNSA0M2MtNDAuNjA2IDcuNTQyLTYzLjEwNi05LjEyNS02Ny41LTUwLTQuNjY4LTI1LjY1Ny0yLjMzNC01MC42NTcgNy03NSA2LjI3Mi0xMi4yNjkgMTYuMTA1LTIwLjEwMiAyOS41LTIzLjUgMzcuNjMyLTEuNDYyIDc1LjI5OS0xLjk2MiAxMTMtMS41di04aC03M2E4NDAuMjcgODQwLjI3IDAgMCAxIC41LTQxYzIuMTM2LTEwLjEzOSA3Ljk2OS0xNy4zMDYgMTcuNS0yMS41IDExLjIyNS0zLjYwOSAyMi41NTgtNi4xMDkgMzQtNy41em0tMjMgMjNjMTIuMTUxLS4wMTUgMTcuMzE4IDUuOTg1IDE1LjUgMTgtMy44OTkgNy45NjktMTAuMDY2IDEwLjQ2OS0xOC41IDcuNS0xMC4wMTUtOS43MzYtOS4wMTUtMTguMjM2IDMtMjUuNXoiIG9wYWNpdHk9Ii45ODQiLz48cGF0aCBkPSJNMzUzLjUgMTc5LjVjMTEuMzM4LS4xNjcgMjIuNjcyIDAgMzQgLjUgMTMuNjA4IDMuNjA0IDIyLjc3NCAxMi4xMDQgMjcuNSAyNS41IDEzLjk3NSAzNy4yNzggMTEuOTc1IDczLjYxMS02IDEwOS00LjY3OSA3LjkxOS0xMS41MTIgMTIuNzUzLTIwLjUgMTQuNWwtMTE3IC41djloNzNjLjQyNSAxMy4zOTQtLjA3NSAyNi43MjctMS41IDQwLTMuNzc2IDguNzc2LTkuOTQyIDE1LjI3Ni0xOC41IDE5LjUtMzEuMDIgMTMuMzU3LTYyLjY4NyAxNS4wMjMtOTUgNS0xNS4zMTktNC4zMjYtMjUuODE5LTEzLjgyNi0zMS41LTI4LjUtLjY2Ny0yOC42NjctLjY2Ny01Ny4zMzMgMC04NiAzLjcwNy0xNC4yMzQgMTIuNTQtMjMuNzM0IDI2LjUtMjguNSAzMC4yOTktMS4yODYgNjAuNjMyLTEuOTUzIDkxLTIgMTkuMDg0LTQuNzUxIDMxLjI1LTE2LjU4NCAzNi41LTM1LjVhMzY0LjI3IDM2NC4yNyAwIDAgMCAxLjUtNDN6bS00NCAxNzhjMTIuODE1Ljk3MiAxNy42NDggNy42MzggMTQuNSAyMC01LjA2OCA3LjM2OC0xMS41NjggOC44NjgtMTkuNSA0LjUtNi4yMjktNi43MjUtNi41NjItMTMuNzI1LTEtMjEgMi4yMjctLjk0MSA0LjIyNy0yLjEwOCA2LTMuNXoiIG9wYWNpdHk9Ii45ODUiLz48L3N2Zz4=)
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
- **FLGBM**
    - **num_leaves**: número de hojas que tendrá el árbol.
    - **min_data_in_leaf**: mínimo cantidad de datos que necesitará un nodo para poder dividirse.
    - **max_depth**: profundidad máxima del árbol.
    - **learning_rate**: tasa de aprendizaje del algoritmo.
    - **n_estimators**: número de clasificadores débiles a construir.
    - **feature_fraction**: proporción de características utilizadas para construir el modelo.
    - **fair_param**: controla la importancia entre la función de pérdida estándar (logloss) del algoritmo, con la función de justicia considerada.

Los objetivos a minimizar durante la experimentación han sido:

- **Inverted G-mean** (gmean_inv): El criterio de media geométrica se define como la raiz del producto de la tasa de verdaderos positivos y la de verdaderos negativos $\sqrt{\text{TPR} \cdot \text{TNR}}$. Al tratarse de un objetivo de minimización, se usará $1-\sqrt{\text{TPR} \cdot \text{TNR}}$.
- **Difference in False Positive Rate** (FPR$_{\text{diff}}$): Resulta de la diferencia entre las probabilidades $|P[p=1|Y=0,A=0]-P[p=1|Y=0,A=1]|$, siendo $p$ el predictor utilizado, $Y$ el atributo a predecir, y $A$ un atributo sensible.


## Results

Los resultados han mostrado que se pueden encontrar soluciones mucho más justas y precisas utilizando los algoritmos empleados que utilizando un árbol de decisión normal.

<div align="center">
  <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_adult.png" width="412px"/> 
  </a>
  <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_compas.png" width="412px"/> 
  </a>
</div>
<div align="center">
  <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_diabetes.png" width="412px"/> 
  </a>
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_dutch.png" width="412px"/> 
  </a>
</div>
<div align="center">
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_german.png" width="412px"/> 
  </a>
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_insurance.png" width="412px"/> 
  </a>
</div>
<div align="center">
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_obesity.png" width="412px"/> 
  </a>
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_parkinson.png" width="412px"/> 
  </a>
</div>
<div align="center">
  <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_ricci.png" width="412px"/> 
  </a>
    <a href="https://github.com/Daalma7/FairTreesAlgorithms/tree/master/other">
    <img src="https://scatter_po_algorithm_student.png" width="412px"/> 
  </a>
</div>


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

