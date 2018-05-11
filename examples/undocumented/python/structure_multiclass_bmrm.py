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

def structure_multiclass_bmrm(fm_train_real=traindat,label_train_multiclass=label_traindat):
	from shogun import MulticlassSOLabels
	from shogun import RealFeatures
	from shogun import SOSVMHelper
	try:
		from shogun import BMRM, PPBMRM, P3BMRM, DualLibQPBMSOSVM
	except ImportError:
		print("At least one of BMRM, PPBMRM, P3BMRM, DualLibQPBMSOSVM not available")
		exit(0)
	from shogun import MulticlassModel, RealNumber

	labels = MulticlassSOLabels(label_train_multiclass)
	features = RealFeatures(fm_train_real.T)

	model = MulticlassModel(features, labels)
	sosvm = DualLibQPBMSOSVM(model, labels, 1.0)

	# BMRM
	sosvm.set_solver(BMRM)
	sosvm.set_verbose(True)
	sosvm.train()

	bmrm_out = sosvm.apply()
	count = 0
	for i in range(bmrm_out.get_num_labels()):
		yi_pred = bmrm_out.get_real_vector("labels_vector")[i]
		if yi_pred == label_train_multiclass[i]:
			count = count + 1

	#print("BMRM: Correct classification rate: %0.2f" % ( 100.0*count/bmrm_out.get_num_labels() ))
	#hp = sosvm.get_helper()
	#print hp.get_primal_values()
	#print hp.get_train_errors()

	# PPBMRM
	w = np.zeros(model.get_dim())
	sosvm.set_w(w)
	sosvm.set_solver(PPBMRM)
	sosvm.set_verbose(True)
	sosvm.train()

	ppbmrm_out = sosvm.apply()
	count = 0
	for i in range(ppbmrm_out.get_num_labels()):
		yi_pred = ppbmrm_out.get_real_vector("labels_vector")[i]
		if yi_pred == label_train_multiclass[i]:
			count = count + 1

	#print("PPBMRM: Correct classification rate: %0.2f" % ( 100.0*count/ppbmrm_out.get_num_labels() ))

	# P3BMRM
	w = np.zeros(model.get_dim())
	sosvm.set_w(w)
	sosvm.set_solver(P3BMRM)
	sosvm.set_verbose(True)
	sosvm.train()

	p3bmrm_out = sosvm.apply()
	count = 0
	for i in range(p3bmrm_out.get_num_labels()):
		yi_pred = p3bmrm_out.get_real_vector("labels_vector")[i]
		if yi_pred == label_train_multiclass[i]:
			count = count + 1

	#print("P3BMRM: Correct classification rate: %0.2f" % ( 100.0*count/p3bmrm_out.get_num_labels() ))
	return bmrm_out, ppbmrm_out, p3bmrm_out

if __name__=='__main__':
	print('SO multiclass model with bundle methods')
	a,b,c=structure_multiclass_bmrm(*parameter_list[0])
