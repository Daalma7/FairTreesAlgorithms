datasets=("adult" "compas" "diabetes" "dutch" "german" "insurance" "obesity" "parkinson" "student" "ricci")
for data in "${datasets[@]}"
do
	for i in {100..109}
	do
		python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=$data bseed=$i nruns=1 model=DT obj=gmean_inv,fpr_diff &
	done
	wait
	python ../HyperparameterOptimization/bin/main.py nind=150 ngen=300 alg=nsga2 dat=$data bseed=100 nruns=10 model=DT obj=gmean_inv,fpr_diff
done
