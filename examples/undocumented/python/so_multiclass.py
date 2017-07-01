#!/usr/bin/env python

import numpy as np

def gen_data(num_classes,num_samples,dim):
	np.random.seed(0)
	covs = np.array([[[0., -1. ], [2.5,  .7]],
			 [[3., -1.5], [1.2, .3]],
			 [[ 2,  0  ], [ .0,  1.5 ]]])
	X = np.r_[np.dot(np.random.randn(num_samples, dim), covs[0]) + np.array([0, 10]),
		  np.dot(np.random.randn(num_samples, dim), covs[1]) + np.array([-10, -10]),
		  np.dot(np.random.randn(num_samples, dim), covs[2]) + np.array([10, -10])];
	Y = np.hstack((np.zeros(num_samples), np.ones(num_samples), 2*np.ones(num_samples)))
	return X, Y

# Number of classes
M = 3
# Number of samples of each class
N = 50
# Dimension of the data
dim = 2

traindat, label_traindat = gen_data(M,N,dim)

parameter_list = [[traindat,label_traindat]]

def so_multiclass (fm_train_real=traindat,label_train_multiclass=label_traindat):
	try:
		from modshogun	import RealFeatures
		from modshogun	import MulticlassModel, MulticlassSOLabels, PrimalMosekSOSVM, RealNumber
	except ImportError:
		print("Mosek not available")
		return

	labels = MulticlassSOLabels(label_train_multiclass)
	features = RealFeatures(fm_train_real.T)

	model = MulticlassModel(features, labels)
	sosvm = PrimalMosekSOSVM(model, labels)
	sosvm.train()

	out = sosvm.apply()
	count = 0
	for i in xrange(out.get_num_labels()):
		yi_pred = RealNumber.obtain_from_generic(out.get_label(i))
		if yi_pred.value == label_train_multiclass[i]:
			count = count + 1

	print("Correct classification rate: %0.2f" % ( 100.0*count/out.get_num_labels() ))

if __name__=='__main__':
	print('SO multiclass')
	so_multiclass(*parameter_list[0])
