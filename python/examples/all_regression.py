#!/usr/bin/env python
"""
Explicit examples on how to use regressions
"""

from sg import sg
from numpy import array, sign
from numpy.random import seed, rand

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
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_regression', 'SVRLIGHT')
	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def libsvr ():
	print 'LibSVR'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_regression', 'LIBSVR')
	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', testdata)
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
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)

	sg('new_regression', 'KRR')
	sg('krr_tau', tau)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', testdata)
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
