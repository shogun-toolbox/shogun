#!/usr/bin/env python
from numpy import *
parameter_list = [[10,3,15,0.9,1,2000,1],[20,4,15,0.9,1,5000,2]]

def classifier_larank_modular (num_vec,num_class,distance,C=0.9,num_threads=1,num_iter=5,seed=1):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import GaussianKernel
	from modshogun import LaRank
	from modshogun import Math_init_random
	
	# reproducible results
	Math_init_random(seed)
	random.seed(seed)

	# generate some training data where each class pair is linearly separable
	label_train=array([mod(x,num_class) for x in range(num_vec)],dtype="float64")
	label_test=array([mod(x,num_class) for x in range(num_vec)],dtype="float64")
	fm_train=array(random.randn(num_class,num_vec))
	fm_test=array(random.randn(num_class,num_vec))
	for i in range(len(label_train)):
		fm_train[label_train[i],i]+=distance
		fm_test[label_test[i],i]+=distance

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	epsilon=1e-5
	labels=MulticlassLabels(label_train)

	svm=LaRank(C, kernel, labels)
	#svm.set_tau(1e-3)
	svm.set_batch_mode(False)
	#svm.io.enable_progress()
	svm.set_epsilon(epsilon)
	svm.train()
	out=svm.apply(feats_test).get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()


if __name__=='__main__':
	print('LaRank')
	[predictions, svm, labels] = classifier_larank_modular(*parameter_list[0])
