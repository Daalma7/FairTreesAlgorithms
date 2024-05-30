# Development of fairness-aware algorithms based on Decision Trees || Master's final project 

This repository contains the work done on the implementation and testing of 3 different multi-objective algorithms, which include methods for achieving a balance between classification and fairness. This

## Brief descrition of developed algorithms
- **FairDT (FDT)**: Modification of the impurity criterion calculation during decision tree training to also consider fairness. The general form it has is: $$(1-\lambda) * \text{gini/entropy} - \lambda * \text{fairness criterion} $$
- **Fair Genetic Pruning (FGP)**: Consideration of the matrix decision tree (largest decision tree which can be built using the available data that perfectly classifies the training set) for a task and pruning it based on objectives considered.
- **FairLGBM (FLGBM)**: Modification of the loss function in the LightGBM algorithm to incorporate fairness.

To see more a detailed explanation of each of this algorithms, refer to the pdf file which describes the whole project.

## Brief description of the experimentation


## Results

## Libraries and dependencies:

- python=3.10.12
- matplotlib
- pandas
- scikit-learn
- pydotplus
- imblearn
- cython=0.29.37
- lightgbm (from the official lightgbm webpage)
- seaborn
- pygmo
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
