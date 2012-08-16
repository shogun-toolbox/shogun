#!/usr/bin/env python

import numpy as np

from shogun.Features 	import RealFeatures
from shogun.Loss     	import HingeLoss
from shogun.Structure	import MulticlassModel, MulticlassSOLabels, RealNumber, DualLibQPBMSOSVM
from shogun.Structure 	import BMRM, PPBMRM, P3BMRM

def gen_data():
	np.random.seed(0)
	covs = np.array([[[0., -1. ], [2.5,  .7]],
		[[3., -1.5], [1.2, .3]],
		[[ 2,  0  ], [ .0,  1.5 ]]])
	X = np.r_[np.dot(np.random.randn(N, dim), covs[0]) + np.array([0, 10]),
			np.dot(np.random.randn(N, dim), covs[1]) + np.array([-10, -10]),
			np.dot(np.random.randn(N, dim), covs[2]) + np.array([10, -10])];
	Y = np.hstack((np.zeros(N), np.ones(N), 2*np.ones(N)))
	return X, Y

# Number of classes
M = 3
# Number of samples of each class
N = 1000
# Dimension of the data
dim = 2

X, y = gen_data()

labels = MulticlassSOLabels(y)
features = RealFeatures(X.T)

model = MulticlassModel(features, labels)
loss = HingeLoss()

lambda_ = 1e1
sosvm = DualLibQPBMSOSVM(model, loss, labels, features, lambda_)

sosvm.set_cleanAfter(10)	# number of iterations that cutting plane has to be inactive for to be removed
sosvm.set_cleanICP(True)	# enables inactive cutting plane removal feature
sosvm.set_TolRel(0.001)		# set relative tolerance
sosvm.set_verbose(True)		# enables verbosity of the solver
sosvm.set_cp_models(16)		# set number of cutting plane models
sosvm.set_solver(BMRM)		# select training algorithm
#sosvm.set_solver(PPBMRM)
#sosvm.set_solver(P3BMRM)

sosvm.train()

out = sosvm.apply()
count = 0.0
for i in xrange(out.get_num_labels()):
	yi_pred = RealNumber.obtain_from_generic(out.get_label(i))
	if yi_pred.value == y[i]:
		count += 1.0

print "Correct classification rate: %0.2f" % ( 100.0*count/out.get_num_labels() )

