from numpy import double
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import *
from shogun.Classifier import *

import util

def _svm (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)
	kargs=util.get_args(input)

	kfun=eval(input['kname']+'Kernel')
	k=kfun(feats['train'], feats['train'], *kargs)
	k.parallel.set_num_threads(input['svmparam_num_threads'])
	l=Labels(double(input['svmparam_labels']))
	svm=SVMLight(input['svmparam_C'], k, l)
	svm.parallel.set_num_threads(input['svmparam_num_threads'])
	svm.set_epsilon(input['svmparam_epsilon'])
	svm.set_tube_epsilon(input['svmparam_tube_epsilon'])
	svm.train()

	if input['svmparam_num_threads']==1:
		check_alphas=max(abs(svm.get_alphas()-input['svmparam_alphas']))
		check_bias=abs(svm.get_bias()-input['svmparam_bias'])
		check_sv=max(abs(svm.get_support_vectors()-input['svmparam_support_vectors']))
	else: # lower accuracy, less checks if parallel
		accuracy=1e-4
		check_alphas=0.
		check_bias=0.
		check_sv=0.

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

