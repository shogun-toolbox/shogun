"""
Test Classifier
"""

from numpy import double
from shogun.Features import Labels
from shogun.Kernel import *
from shogun.Distance import *
from shogun.Classifier import *

import util

def _get_machine (indata, feats):
	if indata['classifier_type']=='kernel':
		kargs=util.get_args(indata, 'kernel_arg')
		kfun=eval(indata['kernel_name']+'Kernel')
		machine=kfun(feats['train'], feats['train'], *kargs)
		machine.parallel.set_num_threads(indata['classifier_num_threads'])
	elif indata['classifier_type']=='knn':
		dargs=util.get_args(indata, 'distance_arg')
		dfun=eval(indata['distance_name'])
		machine=dfun(feats['train'], feats['train'], *dargs)
		machine.parallel.set_num_threads(indata['classifier_num_threads'])
	else:
		machine=None

	return machine

def _get_results (indata, classifier, machine=None, feats=None):
	res={
		'alphas':0,
		'bias':0,
		'sv':0,
		'accuracy':indata['classifier_accuracy'],
	}

	if indata['classifier_num_threads']==1:
		if indata.has_key('classifier_bias'):
			res['bias']=abs(classifier.get_bias()-indata['classifier_bias'])
		if indata.has_key('classifier_alphas'):
			res['alphas']=max(abs(classifier.get_alphas()-indata['classifier_alphas']))
		if indata.has_key('classifier_support_vectors'):
			res['sv']=max(abs(
				classifier.get_support_vectors()-indata['classifier_support_vectors']))
	else: # lower accuracy
		res['accuracy']=1e-4

	if (indata['classifier_type']=='kernel' or
		indata['classifier_type']=='knn'):
		machine.init(feats['train'], feats['test'])

	res['classified']=max(abs(
		classifier.classify().get_labels()-indata['classifier_classified']))

	return res

def _classifier (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	machine=_get_machine(indata, feats)

	# cannot refactor into function, because labels is unrefed otherwise
	fun=eval(indata['name'])
	if indata.has_key('classifier_labels'):
		labels=Labels(double(indata['classifier_labels']))
		if indata['classifier_type']=='kernel':
			classifier=fun(indata['classifier_C'], machine, labels)
		elif indata['classifier_type']=='linear':
			classifier=fun(indata['classifier_C'], feats['train'], labels)
		elif indata['classifier_type']=='knn':
			classifier=fun(indata['classifier_k'], machine, labels)
		elif indata['classifier_type']=='lda':
			classifier=fun(indata['classifier_gamma'], feats['train'], labels)
		elif indata['classifier_type']=='perceptron':
			classifier=fun(feats['train'], labels)
		else:
			return False
	else:
		classifier=fun(indata['classifier_C'], machine)

	classifier.parallel.set_num_threads(indata['classifier_num_threads'])
	if indata['classifier_type']=='linear':
		if indata.has_key('classifier_bias'):
			classifier.set_bias_enabled(True)
		else:
			classifier.set_bias_enabled(False)
	if indata['classifier_type']=='perceptron':
		classifier.set_learn_rate=indata['classifier_learn_rate']
		classifier.set_max_iter=indata['classifier_max_iter']
	if indata.has_key('classifier_epsilon'):
		classifier.set_epsilon(indata['classifier_epsilon'])
	if indata.has_key('classifier_tube_epsilon'):
		classifier.set_tube_epsilon(indata['classifier_tube_epsilon'])
	if indata.has_key('classifier_max_train_time'):
		classifier.set_max_train_time(indata['classifier_max_train_time'])

	classifier.train()

	res=_get_results(indata, classifier, machine, feats)
	return util.check_accuracy(res['accuracy'],
		alphas=res['alphas'], bias=res['bias'], sv=res['sv'],
		classified=res['classified'])

########################################################################
# public
########################################################################

def test (indata):
	return _classifier(indata)

