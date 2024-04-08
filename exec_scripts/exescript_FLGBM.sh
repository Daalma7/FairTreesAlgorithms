python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=adult bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=compas bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=german bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

#FALLO
python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=ricci bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=obesity bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=insurance bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=student bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=diabetes bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=parkinson bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &

python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=dutch bseed=100 nruns=10 model=FLGBM obj=gmean_inv,fpr_diff &
