#!/usr/bin/env python
###########################################################################
# kernel ridge regression
###########################################################################
from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat]]

def regression_least_squares_modular (fm_train=traindat,fm_test=testdat,label_train=label_traindat,tau=1e-6):

	from modshogun import RegressionLabels, RealFeatures
	from modshogun import GaussianKernel
	from modshogun import LeastSquaresRegression

	ls=LeastSquaresRegression(RealFeatures(traindat), RegressionLabels(label_train))
	ls.train()
	out = ls.apply(RealFeatures(fm_test)).get_labels()
	return out,ls

if __name__=='__main__':
	print('LeastSquaresRegression')
	regression_least_squares_modular(*parameter_list[0])
