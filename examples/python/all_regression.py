#!/usr/bin/env python
"""
Explicit examples on how to use regressions
"""

from sg import sg
from numpy import array, sign
from numpy.random import seed, rand

from tools.load import load_features, load_labels
fm_train=load_features('../data/fm_train_real.dat')
fm_test=load_features('../data/fm_test_real.dat')
label_train=load_labels('../data/label_train_oneclass.dat')

###########################################################################
# svm-based
###########################################################################

def svr_light ():
	print 'SVRLight'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2

	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train)

	try:
		sg('new_regression', 'SVRLIGHT')
	except RuntimeError:
		return

	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def libsvr ():
	print 'LibSVR'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2

	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train)
	sg('new_regression', 'LIBSVR')
	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	sg('init_kernel', 'TEST')
	result=sg('classify')

###########################################################################
# misc
###########################################################################

def krr ():
	print 'KRR'

	size_cache=10
	width=2.1
	C=0.017
	tau=1e-6

	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train)

	sg('new_regression', 'KRR')
	sg('krr_tau', tau)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	sg('init_kernel', 'TEST')
	result=sg('classify')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	svr_light()
	libsvr()

	krr()
