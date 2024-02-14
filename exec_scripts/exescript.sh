#python ../GeneticPruning/main.py nind=30 ngen=50 dat=adult bseed=100 nruns=1 obj=accuracy,fpr_diff

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=adult bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr
