cd ~/cvs.II/Genefinder.cvs8/src/svm_cplex

XT=rand(10,100) ;
LT=sign(randn(1,100)) ;

[w,b]=train_svm(XT, LT, 1000)


mcc -m -O all train_svm
