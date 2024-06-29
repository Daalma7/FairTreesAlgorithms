#datasets=("dutch")
datasets=("adult" "compas" "diabetes" "dutch" "german" "insurance" "obesity" "parkinson" "student" "ricci")
for data in "${datasets[@]}"
do
	for i in {100..109}
	do
		python ../FairGeneticPruning/main.py nind=150 ngen=300 dat=$data bseed=$i nruns=1 obj=gmean_inv,fpr_diff &
	done
	wait
done

