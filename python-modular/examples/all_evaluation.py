#!/usr/bin/env python

import numpy
import pylab

from shogun.Features import RealFeatures, SparseRealFeatures, Labels
from shogun.Classifier import SVMOcas
from shogun.Evaluation import *


QUITKEY='q'

def quit (event):
	if event.key==QUITKEY or event.key==QUITKEY.upper():
		pylab.close()

def classify (true_labels):
	num_feats=2
	num_vec=true_labels.get_num_labels()

	data_train=numpy.concatenate(
		(numpy.random.randn(num_feats, num_vec/2)-1,
			numpy.random.randn(num_feats, num_vec/2)+1),
		axis=1)
	realfeat=RealFeatures(data_train)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	C=3.
	svm=SVMOcas(C, feats_train, true_labels)
	svm.train()

	# prepare test data
	multiplier=1.2

	# yields perfect curve without permutation, but random guess curve when
	# shuffled
#	data_test=[]
#	for i in xrange(len(data_train)):
#		data_test.append(numpy.linspace(
#			multiplier*numpy.min(data_train[i]),
#			multiplier*numpy.max(data_train[i]), num_vec))
#		data_test[i]=numpy.random.permutation(data_test[i])
#	data_test=numpy.array(data_test)

	# taken from matplotlib/multiple_svms.py, got issues due to
	# rounding problems. works nicely when round(sqrt(x))^2==x
#	print 'num_vec', num_vec
	num=numpy.sqrt(num_vec)
#	print num
	x1=numpy.linspace(
			multiplier*numpy.min(data_train[0]),
			multiplier*numpy.max(data_train[0]), num)
#	print x1
	x2=numpy.linspace(
			multiplier*numpy.min(data_train[1]),
			multiplier*numpy.max(data_train[1]), num)
	x,y=numpy.meshgrid(x1,x2)
	data_test=numpy.array((numpy.ravel(x), numpy.ravel(y)))
#	print numpy.ravel(x1)
#	print len(data_test[0])

	realfeat=RealFeatures(data_test)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)
	svm.set_features(feats_test)
	return numpy.array(svm.classify().get_labels())

###########################################################################
# performance measures
###########################################################################

def roc():
	num_points=100

	# why does it have to be sorted?
#	true_labels=Labels(numpy.random.rand(num_points).round()*2-1)
#	true_labels=Labels(numpy.sort(numpy.random.rand(num_points).round()*2-1))
	true_labels=Labels(numpy.concatenate(
		(-numpy.ones(num_points/2), numpy.ones(num_points/2))))
	output=Labels(classify(true_labels))
	#print "### true labels\n", true_labels.get_labels()
	#print "### output\n", output.get_labels()


	pm=PerformanceMeasures(true_labels, output)
	points=pm.compute_ROC()
	#print "### ROC points"
	#print points
	points=numpy.array(points).T # for pylab.plot

	pylab.plot(points[0], points[1], 'bo', linewidth=2.)
	pylab.plot([0, 1], [0, 1], 'r-s')
	pylab.axis([0, 1, 0, 1])
	ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
	pylab.xticks(ticks)
	pylab.yticks(ticks)
	pylab.title('ROC of SVMOcas with random examples')
	pylab.xlabel('false positive rate')
	pylab.ylabel('true positive rate')

	pylab.connect('key_press_event', quit)
	pylab.show()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	numpy.random.seed(42)

	roc()
