#!/usr/bin/env python

import sys
from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import NO_NORMALIZATION
from shogun.Classifier import *

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60

STR_ALPHABET='DNA'

STRING_DEGREE=20

WORD_ORDER=3
WORD_GAP=0
WORD_REVERSE=False

##################################################################
## helpers
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

def _kernel_svm(feats, data, name, *args, **kwargs):
	if len(args) < 2:
		print '%s::%s needs at least two variable arguments!' % (_func_name(), kernel)
		return False

	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svm=SVMLight(args[1], k, l) # assumes second vararg is size
	svm.train()
	alphas = svm.get_alphas()

	mats={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'alphas':alphas,
		'labels':labels
	}

	return ['svm_'+name, mats]

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

def weighted_degree_string (feats, data, degree=STRING_DEGREE):
	r=arange(1,degree+1,dtype=double)
	weights = r[::-1]/sum(r)
	params={'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ, 'degree':degree}
	return _kernel(feats, data, 'WeightedDegreeString',
		degree, weights=weights)+[params]

def weighted_degree_position_string (feats, data, degree=STRING_DEGREE):
	shift=ones(LEN_SEQ, dtype=int32)
	params={'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ, 'degree':degree}
	return _kernel(feats, data, 'WeightedDegreePositionString', degree,
		shift)+[params]

def locality_improved_string (feats, data, size=110, length=51, idegree=5, odegree=7):
	params={'size_':size, 'length':length, 'idegree':idegree, 'odegree':odegree}
	return _kernel(feats, data, 'LocalityImprovedString', size, length,
		idegree, odegree)+[params]

def common_word_string (feats, data):
	# ASK: False, NO_NORMALIZATION, 10)
	params={'order':WORD_ORDER, 'gap':WORD_GAP,
		'reverse':str(WORD_REVERSE), 'alphabet':STR_ALPHABET}
	return _kernel(feats, data, 'CommWordString')+[params]

def manhattan_word (feats, data):
	params={'order':WORD_ORDER, 'gap':WORD_GAP,
		'reverse':str(WORD_REVERSE), 'alphabet':STR_ALPHABET}
	return _kernel(feats, data, 'ManhattanWord')+[params]

def hamming_word (feats, data, size=50, width=10, use_sign=False):
	params={'order':WORD_ORDER, 'gap':WORD_GAP,
		'reverse':str(WORD_REVERSE), 'alphabet':STR_ALPHABET,
		'size_':size, 'width_':width, 'use_sign':str(use_sign)}
	return _kernel(feats, data, 'HammingWord', size, width, use_sign)+[params]


##################################################################
## classifiers
##################################################################

def svm_gaussian (feats, data, width, size=10):
	params={'width_':width, 'size_':size}
	try:
		return _kernel_svm(feats, data, 'Gaussian', width, size)+[params]
	except TypeError:
		return False


