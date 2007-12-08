from numpy import double
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import *
from shogun.Distance import *
from shogun.Classifier import *

import util

def _get_machine (input, feats):
	if input['classifier_type']=='kernel':
		kargs=util.get_args(input, 'kernel_arg')
		kfun=eval(input['kernel_name']+'Kernel')
		machine=kfun(feats['train'], feats['train'], *kargs)
		machine.parallel.set_num_threads(input['classifier_num_threads'])
	elif input['classifier_type']=='distance':
		dargs=util.get_args(input, 'distance_arg')
		dfun=eval(input['distance_name'])
		machine=dfun(feats['train'], feats['train'], *dargs)
		machine.parallel.set_num_threads(input['classifier_num_threads'])
	else:
		machine=None

	return machine

def _get_classifier (input, feats):
	fun=eval(input['name'])
	if input.has_key('classifier_labels'):
		labels=Labels(double(input['classifier_labels']))
		if input['classifier_type']=='kernel':
			return fun(input['classifier_C'], kernel, labels)
		elif input['classifier_type']=='linear':
			return fun(input['classifier_C'], feats['train'], labels)
		elif input['classifier_type']=='distance':
			return fun(input['classifier_k'], distance, labels)
		else:
			return False
	else:
		return fun(input['classifier_C'], kernel)

def _get_results (input, classifier, machine=None, feats=None):
	res={
		'alphas':0,
		'bias':0,
		'sv':0,
	}

	if input['classifier_num_threads']==1:
		if input.has_key('classifier_bias'):
			res['bias']=abs(classifier.get_bias()-input['classifier_bias'])
		if input.has_key('classifier_alphas'):
			res['alphas']=max(abs(classifier.get_alphas()-input['classifier_alphas']))
		if input.has_key('classifier_support_vectors'):
			res['sv']=max(abs(classifier.get_support_vectors()-input['classifier_support_vectors']))
	else: # lower accuracy
		accuracy=1e-4

	if (input['classifier_type']=='kernel' or
		input['classifier_type']=='distance'):
		machine.init(feats['train'], feats['test'])

	res['classified']=max(abs(
		classifier.classify().get_labels()-input['classifier_classified']))

	return res

def _classifier (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)

	machine=_get_machine(input, feats)

	# cannot refactor into function, because labels is unrefed otherwise
	fun=eval(input['name'])
	if input.has_key('classifier_labels'):
		labels=Labels(double(input['classifier_labels']))
		if input['classifier_type']=='kernel':
			classifier=fun(input['classifier_C'], machine, labels)
		elif input['classifier_type']=='linear':
			classifier=fun(input['classifier_C'], feats['train'], labels)
		elif input['classifier_type']=='distance':
			classifier=fun(input['classifier_k'], machine, labels)
		else:
			return False
	else:
		classifier=fun(input['classifier_C'], machine)

	classifier.parallel.set_num_threads(input['classifier_num_threads'])
	if input['classifier_type']=='linear':
		if input.has_key('classifier_bias'):
			classifier.set_bias_enabled(True)
		else:
			classifier.set_bias_enabled(False)
	if input.has_key('classifier_epsilon'):
		classifier.set_epsilon(input['classifier_epsilon'])
	if input.has_key('classifier_tube_epsilon'):
		classifier.set_tube_epsilon(input['classifier_tube_epsilon'])

	classifier.train()

	res=_get_results(input, classifier, machine, feats)
	return util.check_accuracy(input['classifier_accuracy'],
		alphas=res['alphas'], bias=res['bias'], sv=res['sv'],
		classified=res['classified'])

########################################################################
# public
########################################################################

def test (input):
	return _classifier(input)

