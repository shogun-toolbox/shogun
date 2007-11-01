#!/usr/bin/env python

import sys
from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import NO_NORMALIZATION
from shogun.Classifier import *

##################################################################
## helpers
##################################################################

def _func_name():
	return sys._getframe(1).f_code.co_name

def _kernel (feats, data, kernel, *args, **kwargs):
	kfun=eval(kernel+'Kernel')
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	km_train=k.get_kernel_matrix()

	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()

	fun = 'test_' + kernel.lower()

	mats={
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}

	return [kernel, fun, mats]

def _kernel_svm(feats, data, kernel, *args, **kwargs):
	if len(args) < 2:
		print '%s::%s needs at least two variable arguments!' % (_func_name(), kernel)
		return False

	kfun=eval(kernel+'Kernel')
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svm=SVMLight(args[1], k, l) # assumes second vararg is size
	svm.train()
	alphas = svm.get_alphas()

	fun = 'test_svm_' + kernel.lower()

	mats={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'alphas':alphas,
		'labels':labels
	}

	return ['svm_'+kernel, fun, mats]

##################################################################
## actual kernels
##################################################################

def gaussian (feats, data, width=1.3, size=10):
	params={'width_':width, 'size_':size}
	return _kernel(feats, data, 'Gaussian', width, size)+[params]

def linear (feats, data, scale=1.0):
	params={'scale':scale}
	return _kernel(feats, data, 'Linear', scale)+[params]

def chi2 (feats, data, size=10):
	params={'size_':size}
	return _kernel(feats, data, 'Chi2', size)+[params]

def sigmoid (feats, data, gamma=1.1, coef0=1.3, size=10):
	params={'size_':size, 'gamma_':gamma, 'coef0':coef0}
	return _kernel(feats, data, 'Sigmoid', size, gamma, coef0)+[params]

def poly (feats, data, inhom=True, use_norm=True, degree=3, size=10):
	params={'size_':size, 'degree':degree, 'inhom':str(inhom),
		'use_norm':str(use_norm)}
	return _kernel(feats, data, 'Poly',
		size, degree, inhom, use_norm)+[params]

def weighted_degree_string (feats, data, seqlen=60, degree=20, alphabet='DNA'):
	r=arange(1,degree+1,dtype=double)
	weights = r[::-1]/sum(r)
	params={'alphabet':alphabet, 'degree':degree, 'seqlen':seqlen}
	return _kernel(feats, data, 'WeightedDegreeString',
		degree, weights=weights)+[params]

def weighted_degree_position_string (feats, data, seqlen=60, degree=20, alphabet='DNA'):
	params={'alphabet':alphabet, 'degree':degree, 'seqlen':seqlen}
	return _kernel(feats, data, 'WeightedDegreePositionString',
		degree, ones(seqlen, dtype=int32))+[params]

def locality_improved_string (feats, data):
#size = 110
#length = 51
#inner_degree = 5
#outer_degree = 7
#k = LocalityImprovedStringKernel(int32(size), length, inner_degree, outer_degree)
	pass

def common_word_string (feats, data, seqlen=60, alphabet='DNA', order=3, gap=0, reverse=False):
	wordfeat_train = StringWordFeatures(feats['train'].get_alphabet());
	wordfeat_train.obtain_from_char(feats['train'], order-1, order, gap, reverse)
	wordfeat_test = StringWordFeatures(feats['test'].get_alphabet());
	wordfeat_test.obtain_from_char(feats['test'], order-1, order, gap, reverse)

	preproc = SortWordString();
	preproc.init(wordfeat_train);
	wordfeat_train.add_preproc(preproc)
	wordfeat_train.apply_preproc()

	preproc = SortWordString();
	preproc.init(wordfeat_test);
	wordfeat_test.add_preproc(preproc)
	wordfeat_test.apply_preproc()

	feats['train'] = wordfeat_train
	feats['test'] = wordfeat_test
	# ASK: False, NO_NORMALIZATION, 10)
	#'seqlen':seqlen, 
	params={'alphabet':alphabet, 'order':order, 'gap':gap,
		'reverse':str(reverse)}
	return _kernel(feats, data, 'CommWordString')+[params]

def hamming_word (feats, data):
#size=50
#width=10
#use_sign=False
#k = HammingWordKernel(wordfeat, wordfeat, size, width, use_sign)
	pass

def manhattan_word (feats, data):
#k = ManhattanWordKernel(wordfeat, wordfeat, size, width)
#km_train=k.get_kernel_matrix()

#k.init(wordfeat,wordtestfeat)
#km_test=k.get_kernel_matrix()

#write_testcase('CommWordStringKernel','test_cws_kernel', {'km_train':km_train ,'km_test':km_test, data_train=matrix(data_train), data_test=matrix(data_test), {'alphabet':'DNA', 'order':order, 'gap':gap, 'reverse':reverse})
	pass


##################################################################
## classifiers
##################################################################

def svm_gaussian (feats, data, width, size=10):
	params={'width_':width, 'size_':size}
	try:
		return _kernel_svm(feats, data, 'Gaussian', width, size)+[params]
	except TypeError:
		return False


