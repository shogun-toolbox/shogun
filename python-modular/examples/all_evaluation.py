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
	pylab.title('ROC of SVMOcas w/ %d random examples'%pm.get_num_labels())
	pylab.xlabel('1 - specificity (false positive rate)')
	pylab.ylabel('sensitivity (true positive rate)')

	points=pm.get_ROC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', linewidth=3.)

	accuracy=pm.get_accuracyROC()
	range_accuracy=numpy.linspace(0, 1, len(accuracy))
	pylab.plot(range_accuracy, accuracy, 'g-')

	# not useful here, hence not plotted
	#error=pm.get_errorROC()

	pylab.plot([0, 1], [0, 1], 'r-')

	auROC=pm.get_auROC()
	aoROC=pm.get_aoROC()
	acc0=pm.get_accuracy0();
	text="auROC = %f\naoROC = %f\naccuracy0 = %f"%(auROC, aoROC, acc0)
	legend=pylab.legend(('ROC', 'accuracy', 'random guess', text),
		loc='lower right')
	texts=legend.get_texts()
	pylab.setp(texts, fontsize='small')

def prc(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)
	pylab.title('PRC of SVMOcas w/ %d random examples'%pm.get_num_labels())
	pylab.xlabel('recall (true positive rate)')
	pylab.ylabel('precision')

	points=pm.get_PRC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', linewidth=3.)

	fmeasure=pm.get_fmeasurePRC()
	range_fmeasure=numpy.linspace(0, 1, len(fmeasure))
	pylab.plot(range_fmeasure, fmeasure, 'g-')

	auPRC=pm.get_auPRC()
	aoPRC=pm.get_aoPRC()
	fmeasure0=pm.get_fmeasure0();
	text="auPRC = %f\naoPRC = %f\nF-measure0 = %f"%(auPRC, aoPRC, fmeasure0)
	legend=pylab.legend(('PRC', 'F-measure', text), loc='lower right')
	texts=legend.get_texts()
	pylab.setp(texts, fontsize='small')

def det(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)
	pylab.title('DET of SVMOcas w/ %d random examples'%pm.get_num_labels())
	pylab.xlabel('log false positive rate')
	pylab.ylabel('log false negative rate')

	points=pm.get_DET()
	points=numpy.array(points).T # for pylab.plot
	pylab.loglog(points[0], points[1], 'b-', linewidth=3.)
	pylab.grid(True)
	pylab.gca().xaxis.grid(True, which='minor')  # minor grid on too

	auDET=pm.get_auDET()
	aoDET=pm.get_aoDET()
	text="auDET=%f\naoDET=%f"%(auDET, aoDET)
	legend=pylab.legend(('DET', text), loc='lower left')
	texts=legend.get_texts()
	pylab.setp(texts, fontsize='small')

def cc_wracc_balance(pm):
	print 'CC at threshold 0:', pm.get_CC0()
	print 'WR accuracy at threshold 0:', pm.get_WRacc0()
	print 'Balance at threshold 0', pm.get_balance0()

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

	roc(pm, 1, 3, 1)
	prc(pm, 1, 3, 2)
	det(pm, 1, 3, 3)
	cc_wracc_balance(pm)

	pylab.connect('key_press_event', quit)
	pylab.show()


