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

def roc(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)

	points=pm.get_ROC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', label='ROC', linewidth=3.)

	accuracy=pm.get_accuracyROC()
	range_accuracy=numpy.linspace(0, 1, len(accuracy))
	pylab.plot(range_accuracy, accuracy, 'g-', label='accuracy')

	# not useful here, hence not plotted
	#error=pm.get_errorROC()

	auROC=pm.get_auROC()
	aoROC=pm.get_aoROC()
	acc0=pm.get_accuracy0();
	text="auROC = %g\naoROC = %f\naccuracy0 = %f"%(auROC, aoROC, acc0)
	pylab.text(.6, .3, text, bbox=dict(fc='white', ec='black', pad=10.))

	pylab.plot([0, 1], [0, 1], 'r-', label='random guess')
	pylab.axis([0, 1, 0, 1])
	ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
	pylab.xticks(ticks)
	pylab.yticks(ticks)
	pylab.title('ROC of SVMOcas with %d random examples/true labels'%pm.get_num_labels())
	pylab.xlabel('1 - specificity (false positive rate)')
	pylab.ylabel('sensitivity (true positive rate)')
	pylab.legend(loc='lower right')

def prc(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)

	points=pm.get_PRC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', label='PRC', linewidth=3.)

	fmeasure=pm.get_fmeasurePRC()
	range_fmeasure=numpy.linspace(0, 1, len(fmeasure))
	pylab.plot(range_fmeasure, fmeasure, 'g-', label='F-measure')

	auPRC=pm.get_auPRC()
	aoPRC=pm.get_aoPRC()
	fmeasure0=pm.get_fmeasure0();
	text="auPRC = %g\naoPRC = %f\nF-measure0 = %f"%(auPRC, aoPRC, fmeasure0)
	pylab.text(.03, .3, text, bbox=dict(fc='white', ec='black', pad=10.))

	pylab.axis([0, 1, 0, 1])
	ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
	pylab.xticks(ticks)
	pylab.yticks(ticks)
	pylab.title('PRC of SVMOcas with %d random examples/true labels'%pm.get_num_labels())
	pylab.xlabel('recall (true positive rate)')
	pylab.ylabel('precision')
	pylab.legend(loc='lower left')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	# fixate 'random' values
	numpy.random.seed(42)

	num_points=500
	true_labels=Labels(numpy.concatenate(
		(-numpy.ones(num_points/2), numpy.ones(num_points/2))))
	output=Labels(classify(true_labels))
	pm=PerformanceMeasures(true_labels, output)

	roc(pm, 1, 2, 1)
	prc(pm, 1, 2, 2)

	pylab.connect('key_press_event', quit)
	pylab.show()


