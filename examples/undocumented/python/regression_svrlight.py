#!/usr/bin/env python
###########################################################################
# svm light based support vector regression
###########################################################################
from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,1.2,1,1e-5,1e-2,1],[traindat,testdat,label_traindat,2.3,0.5,1e-5,1e-6,1]]

def regression_svrlight (fm_train=traindat,fm_test=testdat,label_train=label_traindat, \
				    width=1.2,C=1,epsilon=1e-5,tube_epsilon=1e-2,num_threads=3):


	from shogun import RegressionLabels
	try:
		from shogun import SVRLight
	except ImportError:
		print('No support for SVRLight available.')
		return
	import shogun as sg

	feats_train=sg.features(fm_train)
	feats_test=sg.features(fm_test)

	kernel=sg.kernel("GaussianKernel", log_width=width)

	labels=RegressionLabels(label_train)

	svr=SVRLight(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = svr.apply().get_labels()

	return out, kernel

if __name__=='__main__':
	print('SVRLight')
	regression_svrlight(*parameter_list[0])
