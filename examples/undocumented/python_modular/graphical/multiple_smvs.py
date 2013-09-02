#!/usr/bin/env python
# -*- coding: latin-1 -*-

from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,connect,axis
from numpy import concatenate
from numpy.random import randn
from modshogun import *
from modshogun import *
from modshogun import *
import util

util.set_title('Multiple SVMS')

num_svms=6
width=0.5

svmList = [None]*num_svms
trainfeatList = [None]*num_svms
traindatList = [None]*num_svms
trainlabList = [None]*num_svms
trainlabsList = [None]*num_svms
kernelList = [None]*num_svms

for i in range(num_svms):
	pos=util.get_realdata(True)
	neg=util.get_realdata(False)
	traindatList[i] = concatenate((pos, neg), axis=1)
	trainfeatList[i] = util.get_realfeatures(pos, neg)
	trainlabsList[i] = util.get_labels(True)
	trainlabList[i] = util.get_labels()
	kernelList[i] = GaussianKernel(trainfeatList[i], trainfeatList[i], width)
	svmList[i] = LibSVM(10, kernelList[i], trainlabList[i])

for i in range(num_svms):
	print "Training svm nr. %d" % (i)
	currentSVM = svmList[i]
	currentSVM.train()
	print currentSVM.get_num_support_vectors()
	print "Done."
	x, y, z=util.compute_output_plot_isolines(
		currentSVM, kernelList[i], trainfeatList[i])
	subplot(num_svms/2, 2, i+1)
	pcolor(x, y, z, shading='interp')
	contour(x, y, z, linewidths=1, colors='black', hold=True)
	scatter(traindatList[i][0,:],traindatList[i][1,:], s=20, marker='o', c=trainlabsList[i], hold=True)
	axis('tight')

connect('key_press_event', util.quit)
show()
