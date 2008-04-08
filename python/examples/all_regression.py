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
	use_bias=0
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'set_kernel GAUSSIAN REAL %d %f' % (size_cache, width))
	sg('send_command', 'init_kernel TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('send_command', 'new_svm SVRLIGHT')
	sg('send_command', 'svr_tube_epsilon %f' % tube_epsilon)
	sg('send_command', 'c %f' % C)
	sg('send_command', 'svm_train')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'init_kernel TEST')
	result=sg('svm_classify')

def libsvr ():
	print 'LibSVR'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	use_bias=0
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'set_kernel GAUSSIAN REAL %d %f' % (size_cache, width))
	sg('send_command', 'init_kernel TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('send_command', 'new_svm LIBSVR')
	sg('send_command', 'svr_tube_epsilon %f' % tube_epsilon)
	sg('send_command', 'c %f' % C)
	sg('send_command', 'svm_train')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'init_kernel TEST')
	result=sg('svm_classify')

###########################################################################
# misc
###########################################################################

def krr ():
	print 'KRR'

	return
	# most stuff not implemented

	size_cache=10
	width=2.1
	C=0.017
	tau=1e-6
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=rand(num_feats, num_trainvec)
	testdata=rand(num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'set_kernel GAUSSIAN REAL %d %f' % (size_cache, width))
	sg('send_command', 'init_kernel TRAIN')

	sg('set_labels', 'TRAIN', trainlab)

	return

	sg('send_command', 'new_svm KRR')
	sg('send_command', 'set_tau %f' % tau)
	sg('send_command', 'c %f' % C)
	sg('send_command', 'svm_train')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'init_kernel TEST')
	result=sg('svm_classify')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	svr_light()
	libsvr()

	krr()
