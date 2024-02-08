#File execution


python preprocess.py
: <<'END_COMMENT'
#adult dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=adult var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=smsemoa dat=adult var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=grea dat=adult var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=nsga2 dat=adult var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves

python totalpo.py alg=nsga2,smsemoa,grea dat=adult var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python totalpo.py alg=nsga2 dat=adult var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2 dat=adult var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=adult var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves

#german dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=german var=age bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=smsemoa dat=german var=age bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=grea dat=german var=age bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=nsga2 dat=german var=age bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves

python totalpo.py alg=nsga2,smsemoa,grea dat=german var=age model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python totalpo.py alg=nsga2,smsemoa,grea dat=german var=age model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2 dat=german var=age model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=german var=age model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves

#propublica_recidivism dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=smsemoa dat=propublica_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=grea dat=propublica_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves

python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2 dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves

#propublica_violent_recidivism dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=smsemoa dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=grea dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves

python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_violent_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python totalpo.py alg=nsga2 dat=propublica_violent_recidivism var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2 dat=propublica_violent_recidivism var=race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=propublica_violent_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves

#ricci dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=ricci var=Race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=smsemoa dat=ricci var=Race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=grea dat=ricci var=Race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python main.py nind=200 ngen=300 alg=nsga2 dat=ricci var=Race bseed=100 nruns=10 model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves

python totalpo.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
python totalpo.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2 dat=ricci var=Race model=DT obj=gmean_inv,dem_fpr extra=dem_ppv,num_leaves
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves

#python plots_2.py alg=nsga2,smsemoa,grea dat=adult var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
#python plots_2.py alg=nsga2,smsemoa,grea dat=german var=age model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
#python plots_2.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
#python plots_2.py alg=nsga2,smsemoa,grea dat=propublica_violent_recidivism var=race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
#python plots_2.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=DT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves
'

#LOGISTIC REGRESSION
:'
#adult dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=adult var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=smsemoa dat=adult var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=grea dat=adult var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=nsga2 dat=adult var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=adult var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python totalpo.py alg=nsga2 dat=adult var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr

#german dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=german var=age bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=smsemoa dat=german var=age bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=grea dat=german var=age bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=nsga2 dat=german var=age bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=german var=age model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=german var=age model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr

#propublica_recidivism dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=smsemoa dat=propublica_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=grea dat=propublica_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr


#propublica_violent_recidivism dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=smsemoa dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=grea dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=nsga2 dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=propublica_violent_recidivism var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python totalpo.py alg=nsga2 dat=propublica_violent_recidivism var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr

#ricci dataset with Decision Trees
python main.py nind=200 ngen=300 alg=nsga2 dat=ricci var=Race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=smsemoa dat=ricci var=Race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=grea dat=ricci var=Race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python main.py nind=200 ngen=300 alg=nsga2 dat=ricci var=Race bseed=100 nruns=10 model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python totalpo.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr


python calculatemeasures.py alg=nsga2 dat=adult var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=adult var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2 dat=german var=age model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=german var=age model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2 dat=propublica_recidivism var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=propublica_recidivism var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2 dat=propublica_violent_recidivism var=race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=propublica_violent_recidivism var=race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2 dat=ricci var=Race model=LR obj=gmean_inv,dem_fpr extra=dem_ppv,dem_pnr
python calculatemeasures.py alg=nsga2,smsemoa,grea dat=ricci var=Race model=LR obj=gmean_inv,dem_fpr,dem_ppv,dem_pnr
END_COMMENT


python main.py nind=200 ngen=300 alg=nsga2 dat=adult bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr
#python main.py nind=200 ngen=300 alg=grea dat=german var=age bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves &
#python main.py nind=200 ngen=300 alg=grea dat=propublica_recidivism var=race bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves &
#python main.py nind=200 ngen=300 alg=grea dat=propublica_violent_recidivism var=race bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves &
#python main.py nind=200 ngen=300 alg=grea dat=ricci var=Race bseed=100 nruns=10 model=FDT obj=gmean_inv,dem_fpr,dem_ppv,num_leaves &
