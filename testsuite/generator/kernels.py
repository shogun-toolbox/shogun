#!/usr/bin/env python

from sys import _getframe
from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import FULL_NORMALIZATION
from shogun.Classifier import *

import fileops

klist=open('klist.py', 'r')
KLIST=eval(klist.read())
klist.close()

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60
STR_ALPHABET='DNA'
SIZE_CACHE=10
WORD_ORDER=3
WORD_GAP=0
WORD_REVERSE=False


##################################################################
## private helpers
##################################################################

def _get_params_real ():
	return {}

def _get_params_string ():
	return {'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ}

def _get_params_word ():
	return {'order':WORD_ORDER, 'gap':WORD_GAP,
		'reverse':WORD_REVERSE, 'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ}

def _func_name():
	return _getframe(1).f_code.co_name

def _compute (name, feats, data, *args):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	km_train=k.get_kernel_matrix()
	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()

	params={
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	params.update(eval('_get_params_'+KLIST[name][0]+'()'))
	for i in range(0, len(KLIST[name][2])):
		params[KLIST[name][2][i]]=args[i]

	return [name, params]

def _compute_svm (name, feats, data, C, *args):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	km_train=k.get_kernel_matrix()


	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svm=SVMLight(C, k, l)
	svm.train()
	alphas=svm.get_alphas()
	bias=svm.get_bias()
	support_vectors=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()
	classified=svm.classify().get_labels()

	params={
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'C':C,
		'alphas':alphas,
		'labels':labels,
		'bias':bias,
		'support_vectors':support_vectors,
		'classified':classified
	}
	params.update(eval('_get_params_'+KLIST[name][0]+'()'))
	for i in range(0, len(KLIST[name][2])):
		params[KLIST[name][2][i]]=args[i]

	return [fileops.SVM+name, params]

def _get_data_rand (want_int=False):
	if want_int:
		return {'train':randint(0, 42, (ROWS, LEN_TRAIN)),
			'test':randint(0, 42, (ROWS, LEN_TEST))}
	else:
		return {'train':rand(ROWS, LEN_TRAIN), 'test':rand(ROWS, LEN_TEST)}

def _get_data_dna ():
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

def _get_feats_real (data, sparse=False):
	if sparse:
		return {'train':SparseRealFeatures(data['train']),
			'test':SparseRealFeatures(data['test'])}
	else:
		return {'train':RealFeatures(data['train']),
			'test':RealFeatures(data['test'])}

def _get_feats_string (data, alphabet=DNA):
	feats={'train':StringCharFeatures(alphabet),
		'test':StringCharFeatures(alphabet)}
	feats['train'].set_string_features(data['train'])
	feats['test'].set_string_features(data['test'])

	return feats

def _get_feats_word (data, alphabet=DNA, order=WORD_ORDER,
	gap=WORD_GAP, reverse=WORD_REVERSE):

	feats={}

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['train'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat,
		WORD_ORDER-1, WORD_ORDER, WORD_GAP, WORD_REVERSE)
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['test'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat,
		WORD_ORDER-1, WORD_ORDER, WORD_GAP, WORD_REVERSE)
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return feats

##################################################################
## run funcs
##################################################################

def _run_realfeats ():
	data=_get_data_rand()
	feats=_get_feats_real(data)

#FIXME C++: optional size
	fileops.write(_compute('Chi2', feats, data, 10))
	fileops.write(_compute('Gaussian', feats, data, 1.3))
	fileops.write(_compute('Linear', feats, data, 1.))
	fileops.write(_compute('Poly', feats, data, 3, True, True))
	#fileops.write(_compute('Poly', feats, data, 3, False, True))
	#fileops.write(_compute('Poly', feats, data, 3, True, False))
	#fileops.write(_compute('Poly', feats, data, 3, False, False))
#FIXME C++: optional size
	fileops.write(_compute('Sigmoid', feats, data, 10, 1.1, 1.3))
	fileops.write(_compute('Sigmoid', feats, data, 10, 0.5, 0.7))

	fileops.write(_compute_svm('Gaussian', feats, data, .017, 1.5))
	#fileops.write(_compute_svm('Gaussian', feats, data, .23, 1.5))

def _run_stringfeats ():
	data=_get_data_dna()
	feats=_get_feats_string(data)

	#fileops.write(_compute('FixedDegreeString', feats, data, 3))
	#fileops.write(_compute('LinearString', feats, data, 1.))
	#fileops.write(_compute('LocalAlignmentString', feats, data))
	#fileops.write(_compute('PolyMatchString', feats, data, 3, False, True))
	#fileops.write(_compute('SimpleLocalityImprovedString', feats, data, 51, 7, 5))

	r=arange(1,20+1, dtype=double)
	weights=r[::-1]/sum(r)
	fileops.write(_compute('WeightedDegreeString', feats, data, 20, 0, True, False, 1, -1, weights))
	shift=ones(LEN_SEQ, dtype=int32)
	fileops.write(_compute('WeightedDegreePositionString', feats, data, 20, shift, True, 0, 1))

	# buggy:
	#fileops.write(_compute('LocalityImprovedString', feats, data. 51, 5, 7))


def _run_wordfeats ():
	data=_get_data_dna()
	feats=_get_feats_word(data)

	fileops.write(_compute('CommWordString', feats, data, False, FULL_NORMALIZATION))
	#fileops.write(_compute('HammingWord', feats, data, 10, False))
	#fileops.write(_compute('LinearWord', feats, data, True, 1.))
	#fileops.write(_compute('ManhattanWord', feats, data, 0))
	fileops.write(_compute('WeightedCommWordString', feats, data, False, FULL_NORMALIZATION))
	
def _run_sparsefeats ():
	data={'train':23, 'test':42}
	feats=_get_feats_real(data, sparse=True)

	# floating point exception not within Python
	#fileops.write(sparse_linear(feats, data))
	#fileops.write(sparse_poly(feats, data))
	#fileops.write(sparse_gaussian(feats, data))

def _run_feats_simpleword ():
	data=_get_data_dna()
	#data=_get_data_rand(want_int=True)
	print data
	#feats=_get_feats_simpleword(data)

	# floating exception...
	#fileops.write(linear_word(feats, data))


def run ():
	_run_realfeats()
	_run_stringfeats()
	_run_wordfeats()
	#_run_sparsefeats()
	#_run_feats_simpleword()


















##################################################################
## actual kernels
##################################################################

def sparse_gaussian (feats, data, width=1.3, size=SIZE_CACHE):
	params={'width_':width}
	params.update(_get_params_real(size))
	return _kernel('SparseGaussian', feats, data, size, width)+[params]

def sparse_linear (feats, data, scale=1.0, size=SIZE_CACHE):
	params={'scale':scale}
	params.update(_get_params_real(size))
	return _kernel('SparseLinear', feats, data, size, scale)+[params]

def sparse_poly (feats, data, degree=3,
	inhomogene=True, use_normalization=True, size=SIZE_CACHE):

	params={'degree':degree, 'inhomogene':inhomogene,
		'use_normalization':use_normalization}
	params.update(_get_params_real(size))
	return _kernel('SparsePoly', feats, data, size, degree, inhomogene,
		use_normalization)+[params]

##################################################################
## classifiers
##################################################################

def svm_gaussian (feats, data, multiplier_labels=1, width=1.5, size=SIZE_CACHE):
	params={'width_':width, 'multiplier_labels':multiplier_labels}
	params.update(_get_params_real(size))
	return _kernel_svm('Gaussian', feats, data, multiplier_labels, size,
		width)+[params]


