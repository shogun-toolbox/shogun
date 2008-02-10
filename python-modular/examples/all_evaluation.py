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

	data_test=numpy.concatenate(
		(numpy.random.randn(num_feats, num_vec/2)-1,
			numpy.random.randn(num_feats, num_vec/2)+1),
		axis=1)
	realfeat=RealFeatures(data_test)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)
	svm.set_features(feats_test)

	return numpy.array(svm.classify().get_labels())

###########################################################################
# performance measures
###########################################################################

def roc():
	num_points=500

	true_labels=Labels(numpy.concatenate(
		(-numpy.ones(num_points/2), numpy.ones(num_points/2))))
	output=Labels(classify(true_labels))

	pm=PerformanceMeasures(true_labels, output)
	points=pm.compute_ROC()
	points=numpy.array(points).T # for pylab.plot

	pylab.plot(points[0], points[1], 'b.', linewidth=2.)
	pylab.plot([0, 1], [0, 1], 'r-s')
	pylab.axis([0, 1, 0, 1])
	ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
	pylab.xticks(ticks)
	pylab.yticks(ticks)
	pylab.title('ROC of SVMOcas with random examples/true labels')
	pylab.xlabel('false positive rate')
	pylab.ylabel('true positive rate')

	pylab.connect('key_press_event', quit)
	pylab.show()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	# fixate 'random' values
	numpy.random.seed(42)

	roc()
