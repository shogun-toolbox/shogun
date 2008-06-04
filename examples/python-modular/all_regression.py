#!/usr/bin/env python
"""
Explicit examples on how to use regressions
"""

from numpy import array
from numpy.random import seed, rand
from shogun.Features import Labels, RealFeatures
from shogun.Kernel import GaussianKernel
from shogun.Regression import *

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train=lm.load_numbers('../data/fm_train_real.dat')
fm_test=lm.load_numbers('../data/fm_test_real.dat')
label_train=lm.load_labels('../data/label_train_oneclass.dat')

###########################################################################
# svm-based
###########################################################################

def svr_light ():
	print 'SVRLight'

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=3
	labels=Labels(label_train)

	try:
		svr=SVRLight(C, epsilon, kernel, labels)
	except NameError:
		print 'No support for SVRLight available.'
		return

	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train()

	kernel.init(feats_train, feats_test)
	svr.classify().get_labels()

def libsvr ():
	print 'LibSVR'

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	labels=Labels(label_train)

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

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	width=0.8
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.42
	tau=1e-6
	num_threads=1
	labels=Labels(label_train)

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
