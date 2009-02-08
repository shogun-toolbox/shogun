#!/usr/bin/env python

import numpy
import pylab

from shogun.Features import RealFeatures, SparseRealFeatures, Labels
from shogun.Classifier import SVMOcas
from shogun.Evaluation import *
import util


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
	pylab.title('ROC')
	pylab.xlabel('1 - specificity (false positive rate)')
	pylab.ylabel('sensitivity (true positive rate)')

	points=pm.get_ROC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', linewidth=3.)

	# random guess
	pylab.plot([0, 1], [0, 1], 'r-')

	auROC=pm.get_auROC()
	aoROC=pm.get_aoROC()
	text="auROC = %f\naoROC = %f"%(auROC, aoROC)
	legend=pylab.legend(('ROC', 'random guess', text), loc='lower right')
	texts=legend.get_texts()
	pylab.setp(texts, fontsize='small')

def prc(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)
	pylab.title('PRC')
	pylab.xlabel('recall (true positive rate)')
	pylab.ylabel('precision')

	points=pm.get_PRC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', linewidth=3.)

	auPRC=pm.get_auPRC()
	aoPRC=pm.get_aoPRC()
	text="auPRC = %f\naoPRC = %f"%(auPRC, aoPRC)
	legend=pylab.legend(('PRC', text), loc='lower right')
	texts=legend.get_texts()
	pylab.setp(texts, fontsize='small')

def det(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)
	pylab.title('DET')
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

def measures(pm, numrows, numcols, fignum):
	pylab.subplot(numrows, numcols, fignum)
	pylab.title('CC / WRAcc / F-Measure / Error / Accuracy / BAL')
	pylab.xlabel('output')
	pylab.ylabel('measure')

	points=pm.get_all_CC()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'b-', label='CC', linewidth=3.)
	
	points=pm.get_all_WRAcc()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'g-', label='WRAcc', linewidth=3.)

	points=pm.get_all_fmeasure()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'm-', label='F-measure', linewidth=3.)

	points=pm.get_all_error()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'c-', label='error', linewidth=3.)

	points=pm.get_all_accuracy()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'k-', label='accuracy', linewidth=3.)

	points=pm.get_all_BAL()
	points=numpy.array(points).T # for pylab.plot
	pylab.plot(points[0], points[1], 'r-', label='BAL', linewidth=3.)


	pylab.legend(loc='upper left')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	# fixate 'random' values
	numpy.random.seed(42)

	num_points=1000
	true_labels=Labels(numpy.concatenate(
		(-numpy.ones(num_points/2), numpy.ones(num_points/2))))
	output=Labels(classify(true_labels))
	pm=PerformanceMeasures(true_labels, output)

	roc(pm, 2, 2, 1)
	prc(pm, 2, 2, 2)
	det(pm, 2, 2, 3)
	measures(pm, 2, 2, 4)

	title='SVMOCas with %d random examples'%num_points
	try:
		pylab.get_current_fig_manager().window.set_title(title)
	except AttributeError:
		pylab.get_current_fig_manager().window.setCaption(title)
	except AttributeError:
		pylab.get_current_fig_manager().window.title(title)
	except AttributeError:
		pass

	pylab.connect('key_press_event', util.quit)
	pylab.show()


