#!/usr/bin/env python
from numpy import *
parameter_list = [[10,3,15,2.1,1,1e-5,1],[20,4,15,2.2,2,1e-5,2]]

def classifier_multiclassocas (num_vec=10,num_class=3,distance=15,width=2.1,C=1,epsilon=1e-5,seed=1):
	from shogun import MulticlassLabels
	import shogun as sg
	try:
		sg.create_machine("MulticlassOCAS")
	except ImportError:
		print("MulticlassOCAS not available")
		return

	# reproducible results
	random.seed(seed)

	# generate some training data where each class pair is linearly separable
	label_train=array([mod(x,num_class) for x in range(num_vec)],dtype="float64")
	label_test=array([mod(x,num_class) for x in range(num_vec)],dtype="float64")
	fm_train=array(random.randn(num_class,num_vec))
	fm_test=array(random.randn(num_class,num_vec))
	for i in range(len(label_train)):
		fm_train[int(label_train[i]),i]+=distance
		fm_test[int(label_test[i]),i]+=distance

	feats_train=sg.create_features(fm_train)
	feats_test=sg.create_features(fm_test)

	labels=sg.create_labels(label_train)

	classifier = sg.create_machine("MulticlassOCAS", labels=labels, C=C)
	classifier.train(feats_train)

	out = classifier.apply(feats_test).get("labels")
	#print label_test
	#print out
	return out,classifier

if __name__=='__main__':
	print('MulticlassOCAS')
	classifier_multiclassocas(*parameter_list[0])
