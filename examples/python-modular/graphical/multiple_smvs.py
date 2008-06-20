#!/usr/bin/env python
# -*- coding: latin-1 -*-

from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,connect,axis
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
import util

util.set_title('Multiple SVMS')

num_svms=6
num_dat=100
width=0.5

svmList = [None]*num_svms
trainfeatList = [None]*num_svms
traindatList = [None]*num_svms
trainlabList = [None]*num_svms
trainlabsList = [None]*num_svms
kernelList = [None]*num_svms

for i in range(num_svms):
	traindatList[i] = concatenate((randn(2,num_dat)-1,randn(2,num_dat)+1),axis=1)
	trainfeatList[i] = RealFeatures(traindatList[i])
	trainlabList[i] = Labels(concatenate((-ones(num_dat), ones(num_dat))))
	trainlabsList[i] = concatenate((-ones(num_dat), ones(num_dat)))
	kernelList[i] = GaussianKernel(trainfeatList[i], trainfeatList[i], width)
	svmList[i] = LibSVM(10, kernelList[i], trainlabList[i])

for i in range(num_svms):
	print "Training svm nr. %d" % (i)
	currentSVM = svmList[i]
	currentSVM.train()
	print currentSVM.get_num_support_vectors()
	print "Done."
	x1=linspace(1.2*min(traindatList[i][0]),1.2*max(traindatList[i][0]), 50)
	x2=linspace(1.2*min(traindatList[i][1]),1.2*max(traindatList[i][1]), 50)
	x,y=meshgrid(x1,x2);
	testdat=RealFeatures(array((ravel(x), ravel(y))))
	kernelList[i].init(trainfeatList[i], testdat);
	l = currentSVM.classify()
	z = currentSVM.classify().get_labels().reshape((50,50))
	subplot(num_svms/2,2,i+1)
	pcolor(x, y, z, shading='interp')
	contour(x, y, z, linewidths=1, colors='black', hold=True)
	scatter(traindatList[i][0,:],traindatList[i][1,:], s=20, marker='o', c=trainlabsList[i], hold=True)
	axis('tight')

connect('key_press_event', util.quit)
show()
