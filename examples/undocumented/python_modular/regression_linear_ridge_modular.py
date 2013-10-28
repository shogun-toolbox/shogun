#!/usr/bin/env python
###########################################################################
# linear ridge regression
###########################################################################
from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,1e-6],[traindat,testdat,label_traindat,100]]

def regression_linear_ridge_modular (fm_train=traindat,fm_test=testdat,label_train=label_traindat,tau=1e-6):

	from modshogun import RegressionLabels, RealFeatures
	from modshogun import LinearRidgeRegression

	rr=LinearRidgeRegression(tau, RealFeatures(traindat), RegressionLabels(label_train))
	rr.train()
	out = rr.apply(RealFeatures(fm_test)).get_labels()
	return out,rr

if __name__=='__main__':
	print('LinearRidgeRegression')
	regression_linear_ridge_modular(*parameter_list[0])
