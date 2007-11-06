#!/usr/bin/env python

import sys
from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import FULL_NORMALIZATION
from shogun.Classifier import *

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60
STR_ALPHABET='DNA'
SIZE_CACHE=10
STRING_DEGREE=20
WORD_ORDER=3
WORD_GAP=0
WORD_REVERSE=False

##################################################################
## private helpers
##################################################################

def _get_params_real (size):
	return {'size_':size}

def _get_params_string (size):
	return {'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ, 'size_':size}

def _get_params_word (size):
	return {'order':WORD_ORDER, 'gap':WORD_GAP,
		'reverse':WORD_REVERSE, 'alphabet':STR_ALPHABET, 'size_':size}

def _func_name():
	return sys._getframe(1).f_code.co_name

def _kernel (feats, data, name, *args, **kwargs):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	km_train=k.get_kernel_matrix()

	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()

	mats={
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}

	return [name, mats]

def _kernel_svm(feats, data, name, multiplier_labels=1, *args, **kwargs):
	if len(args)<2:
		print '%s::%s needs at least two variable arguments!' % (_func_name(), name)
		return False

	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	km_train=k.get_kernel_matrix()

	num_vec=feats['train'].get_num_vectors();
	labels=(rand(num_vec).round()*2-1)*multiplier_labels
	l=Labels(labels)
	svm=SVMLight(args[1], k, l) # assumes second vararg is size
	svm.train()
	alphas=svm.get_alphas()
	bias=svm.get_bias()
	support_vectors=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()
	classified=svm.classify().get_labels()

	output={
		'data_train':matrix(data['train']),
		'km_train':km_train,
		'data_test':matrix(data['test']),
		'km_test':km_test,
		'alphas':alphas,
		'labels':labels,
		'bias':bias,
		'support_vectors':support_vectors,
		'classified':classified
	}

	return ['svm_'+name, output]

##################################################################
## public helpers
##################################################################

def get_data_rand ():
	return {'train':rand(ROWS, LEN_TRAIN), 'test':rand(ROWS, LEN_TEST)}

def get_data_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	train=[]
	test=[]

	for i in range(LEN_TRAIN):
		str1=[]
		str2=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		train.append(''.join(str1))
	test.append(''.join(str2))
	
	for i in range(LEN_TEST-LEN_TRAIN):
		str1=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
	test.append(''.join(str1))

	return {'train': train, 'test': test}

def get_feats_real (data):
	return {'train':RealFeatures(data['train']),
		'test':RealFeatures(data['test'])}

def get_feats_string (data, alphabet=DNA):
	feats={'train':StringCharFeatures(alphabet),
		'test':StringCharFeatures(alphabet)}
	feats['train'].set_string_features(data['train'])
	feats['test'].set_string_features(data['test'])

	return feats

def get_feats_word (data, alphabet=DNA, order=3, gap=0, reverse=False):
	feats={}

	stringfeat=StringCharFeatures(alphabet)
	stringfeat.set_string_features(data['train'])
	wordfeat=StringWordFeatures(stringfeat.get_alphabet());
	wordfeat.obtain_from_char(stringfeat, WORD_ORDER-1, WORD_ORDER,
		WORD_GAP, WORD_REVERSE)
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	stringfeat=StringCharFeatures(alphabet)
	stringfeat.set_string_features(data['test'])
	wordfeat=StringWordFeatures(stringfeat.get_alphabet());
	wordfeat.obtain_from_char(stringfeat, WORD_ORDER-1, WORD_ORDER,
		WORD_GAP, WORD_REVERSE)
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return feats

##################################################################
## actual kernels
##################################################################

def gaussian (feats, data, width=1.3, size=SIZE_CACHE):
	params={'width_':width}
	params.update(_get_params_real(size))
	return _kernel(feats, data, 'Gaussian', width, size)+[params]

def linear (feats, data, scale=1.0, size=SIZE_CACHE):
	params={'scale':scale}
	params.update(_get_params_real(size))
	return _kernel(feats, data, 'Linear', scale, size)+[params]

def chi2 (feats, data, size=SIZE_CACHE):
	params=_get_params_real(size)
	return _kernel(feats, data, 'Chi2', size)+[params]

def sigmoid (feats, data, gamma=1.1, coef0=1.3, size=SIZE_CACHE):
	params={'gamma_':gamma, 'coef0':coef0}
	params.update(_get_params_real(size))
	return _kernel(feats, data, 'Sigmoid', size, gamma, coef0)+[params]

def poly (feats, data, degree=3,
	inhomogene=True, use_normalization=True, size=SIZE_CACHE):

	params={'degree':degree, 'inhomogene':inhomogene,
		'use_normalization':use_normalization}
	params.update(_get_params_real(size))
	return _kernel(feats, data, 'Poly', degree, inhomogene, use_normalization,
		size)+[params]

def weighted_degree_string (feats, data, degree=STRING_DEGREE,
	max_mismatch=0, use_normalization=True, block_computation=False,
	mkl_stepsize=1, which_degree=-1, size=SIZE_CACHE):

	r=arange(1,STRING_DEGREE+1,dtype=double)
	weights=r[::-1]/sum(r)

	params={'degree':degree, 'max_mismatch':max_mismatch,
		'use_normalization':use_normalization,
		'block_computation':block_computation,
		'mkl_stepsize':mkl_stepsize, 'which_degree':which_degree,
		'weights':weights}
	params.update(_get_params_string(size))
	return _kernel(feats, data, 'WeightedDegreeString',
		degree, max_mismatch, use_normalization, block_computation,
		mkl_stepsize, which_degree, weights, size)+[params]

def weighted_degree_position_string (feats, data, degree=STRING_DEGREE,
	use_normalization=True, max_mismatch=0, mkl_stepsize=1,
	size=SIZE_CACHE):

	shifts=ones(LEN_SEQ, dtype=int32)
	params={'degree':degree, 'use_normalization':use_normalization,
		'max_mismatch':max_mismatch, 'mkl_stepsize':mkl_stepsize}
	params.update(_get_params_string(size))
	return _kernel(feats, data, 'WeightedDegreePositionString', degree,
		shifts, use_normalization, max_mismatch, mkl_stepsize, size)+[params]

def locality_improved_string (feats, data,
	length=51, inner_degree=5, outer_degree=7, size=SIZE_CACHE):

	params={'length':length, 'inner_degree':inner_degree,
		'outer_degree':outer_odegree}
	params.update(_get_params_string(size))
	return _kernel(feats, data, 'LocalityImprovedString', size, length,
		inner_degree, outer_degree)+[params]

def common_word_string (feats, data,
	use_sign=False, normalization=FULL_NORMALIZATION, size=SIZE_CACHE):

	params={'use_sign':use_sign, 'normalization':normalization}
	params.update(_get_params_word(size))
	return _kernel(feats, data, 'CommWordString', use_sign, normalization,
		size)+[params]

def manhattan_word (feats, data, width=0, size=SIZE_CACHE):
	params={'width_':width}
	params.update(_get_params_word(size))
	return _kernel(feats, data, 'ManhattanWord', size, width)+[params]

def hamming_word (feats, data, width=10, use_sign=False, size=SIZE_CACHE):
	params={'width_':width, 'use_sign':use_sign}
	params.update(_get_params_word(size))
	return _kernel(feats, data, 'HammingWord', size, width, use_sign)+[params]


##################################################################
## classifiers
##################################################################

def svm_gaussian (feats, data, multiplier_labels=1, width=1.5, size=SIZE_CACHE):
	params={'width_':width, 'multiplier_labels':multiplier_labels}
	params.update(_get_params_real(size))
	try:
		return _kernel_svm(feats, data, 'Gaussian', multiplier_labels, width,
			size)+[params]
	except TypeError:
		return False


