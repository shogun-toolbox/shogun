#!/usr/bin/env python
"""
Explicit examples on how to use regressions
"""

from numpy import array
from numpy.random import seed, rand
from shogun.Features import Labels, RealFeatures
from shogun.Kernel import GaussianKernel
from shogun.Regression import *

###########################################################################
# svm-based
###########################################################################

def svr_light ():
	print 'SVRLight'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=3
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svr=SVRLight(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train()

	kernel.init(feats_train, feats_test)
	svr.classify().get_labels()

def libsvr ():
	print 'LibSVR'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svr=LibSVR(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train()

	kernel.init(feats_train, feats_test)
	svr.classify().get_labels()

###########################################################################
# misc
###########################################################################

def krr ():
	print 'KRR'

	rows=13
	data=rand(rows, 12)
	feats_train=RealFeatures(data)
	data=rand(rows, 19)
	feats_test=RealFeatures(data)
	width=0.8
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.42
	tau=1e-6
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	krr=KRR(tau, kernel, labels)
	krr.parallel.set_num_threads(num_threads)
	krr.train()

	kernel.init(feats_train, feats_test)
	krr.classify().get_labels()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	svr_light()
	libsvr()

	krr()
