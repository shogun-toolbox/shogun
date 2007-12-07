from numpy import double
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import *
from shogun.Classifier import *

import util

def _svm (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)
	type=input['svmparam_type']

	if type=='kernel':
		kargs=util.get_args(input)
		kfun=eval(input['kname']+'Kernel')
		k=kfun(feats['train'], feats['train'], *kargs)
		k.parallel.set_num_threads(input['svmparam_num_threads'])

	svmfun=eval(input['name'])

	if input['svmparam_labels'] is not None:
		lab=Labels(double(input['svmparam_labels']))
		if type=='linear':
			svm=svmfun(input['svmparam_C'], feats['train'], lab)
		else:
			svm=svmfun(input['svmparam_C'], k, lab)
	else:
		svm=svmfun(input['svmparam_C'], k)

	svm.parallel.set_num_threads(input['svmparam_num_threads'])
	svm.set_epsilon(input['svmparam_epsilon'])

	if type=='linear':
		if input.has_key('svmparam_bias'):
			svm.set_bias_enabled(True)
		else:
			svm.set_bias_enabled(False)
	else:
		svm.set_tube_epsilon(input['svmparam_tube_epsilon'])

	svm.train()

	check_alphas=0
	check_bias=0
	check_sv=0
	if input['svmparam_num_threads']==1:
		if type=='linear':
			if input.has_key('svmparam_bias'):
				check_bias=abs(svm.get_bias()-input['svmparam_bias'])
		else:
			check_alphas=max(abs(svm.get_alphas()-input['svmparam_alphas']))
			check_bias=abs(svm.get_bias()-input['svmparam_bias'])
			check_sv=max(abs(svm.get_support_vectors()-input['svmparam_support_vectors']))
	else: # lower accuracy
		accuracy=1e-4

	if type=='kernel':
		k.init(feats['train'], feats['test'])

	check_classified=max(abs(svm.classify().get_labels()-input['svmparam_classified']))

	return util.check_accuracy(input['svmparam_accuracy'],
		alphas=check_alphas, bias=check_bias, sv=check_sv,
		classified=check_classified)

########################################################################
# public
########################################################################

def test (input):
	return _svm(input)

