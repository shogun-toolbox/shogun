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

		if indata['kernel_name']=='Linear':
			machine.set_normalizer(AvgDiagKernelNormalizer(-1))
			machine.init(feats['train'], feats['train'])

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

	if indata.has_key('classifier_bias'):
		res['bias']=abs(classifier.get_bias()-indata['classifier_bias'])
	if indata.has_key('classifier_alphas'):
		res['alphas']=max(abs(classifier.get_alphas()- \
			indata['classifier_alphas']))
	if indata.has_key('classifier_support_vectors'):
		res['sv']=max(abs(classifier.get_support_vectors()- \
			indata['classifier_support_vectors']))

	ctype=indata['classifier_type']
	if ctype=='kernel' or ctype=='knn':
		machine.init(feats['train'], feats['test'])
	elif ctype=='linear' or ctype=='perceptron' or ctype=='lda':
		classifier.set_features(feats['test'])

	res['classified']=max(abs(
		classifier.classify().get_labels()-indata['classifier_classified']))

	return res

def _classifier (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	machine=_get_machine(indata, feats)

	try:
		fun=eval(indata['name'])
	except NameError, e:
		print "%s is disabled/unavailable!"%indata['name']
		return False

	# cannot refactor into function, because labels is unrefed otherwise
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
		try:
			classifier.set_epsilon(indata['classifier_epsilon'])
		except AttributeError:
			pass
	if indata.has_key('classifier_tube_epsilon'):
		classifier.set_tube_epsilon(indata['classifier_tube_epsilon'])
	if indata.has_key('classifier_max_train_time'):
		classifier.set_max_train_time(indata['classifier_max_train_time'])
	if indata.has_key('classifier_linadd_enabled'):
		classifier.set_linadd_enabled(indata['classifier_linadd_enabled'])
	if indata.has_key('classifier_batch_enabled'):
		classifier.set_batch_computation_enabled(indata['classifier_batch_enabled'])

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

