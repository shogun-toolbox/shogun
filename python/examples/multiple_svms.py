#!/usr/bin/env python
# -*- coding: latin-1 -*-

from numpy import array,concatenate,ones
from numpy.random import randn
from shogun.Features import *
from shogun.SVM import *
from shogun.Kernel import *


svmList = [None]*20
traindatList = [None]*20
trainlabList = [None]*20
kernelList = [None]*20

for i in range(20):
	traindatList[i] = RealFeatures(randn(5,200))
	trainlabList[i] = Labels(concatenate((-ones(100), ones(100))))
	kernelList[i] = GaussianKernel(traindatList[i], traindatList[i], 10,1)
	svmList[i] = SVMLight(10, kernelList[i], trainlabList[i])

for i in range(20):
	print "Training svm nr. %d" % (i)
	currentSVM = svmList[i]
	print "Done."
	currentSVM.train()
