#!/usr/bin/env python

from sys import _getframe
from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Distance import *
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

def _func_name():
	return _getframe(1).f_code.co_name

def _get_params_global (type=''):
	if type=='String' or type=='Byte' or type=='Char':
		return {'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ}
	elif type=='Wordstring':
		return {'order':WORD_ORDER, 'gap':WORD_GAP,
			'reverse':WORD_REVERSE, 'alphabet':STR_ALPHABET, 'seqlen':LEN_SEQ}
	else:
		return {}

##################################################################
## compute/kernel funcs
##################################################################

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
	params.update(_get_params_global(KLIST[name][0]))
	for i in range(0, len(args)):
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
	params.update(_get_params_global(KLIST[name][0]))
	for i in range(0, len(args)):
		params[KLIST[name][2][i]]=args[i]

	return [fileops.SVM+name, params]

##################################################################
## feats and data funcs
##################################################################

def _get_data_rand (dattype=False, rows=ROWS):
	if dattype:
		# randint does not understand arg dtype
		train=randint(0, 42, (rows, LEN_TRAIN))
		test=randint(0, 42, (rows, LEN_TEST))
		return {'train':train.astype(dattype), 'test':test.astype(dattype)}
	else:
		return {'train':rand(rows, LEN_TRAIN), 'test':rand(rows, LEN_TEST)}

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

def _get_feats_simple (type, data, alphabet=DNA, sparse=False):
	if type=='Byte' or type=='Char':
		train=eval(type+"Features(data['train'], alphabet)")
		test=eval(type+"Features(data['test'], alphabet)")
	else:
		train=eval(type+"Features(data['train'])")
		test=eval(type+"Features(data['test'])")

	if sparse:
		sparse_train=eval('Sparse'+type+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('Sparse'+type+'Features()')
		sparse_test.obtain_from_simple(test)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':train, 'test':test}

def _get_feats_string (type, data, alphabet=DNA):
	train=eval('String'+type+"Features(alphabet)")
	train.set_string_features(data['train'])
	test=eval('String'+type+"Features(alphabet)")
	test.set_string_features(data['test'])

	return {'train':train, 'test':test}

def _get_feats_wordstring (data, alphabet=DNA, order=WORD_ORDER,
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

# segfaults
def _run_auc ():
	sk=GaussianKernel(10, 1.5)
	data=_get_data_rand(dattype=ushort, rows=2)
	feats=_get_feats_simple('Word', data)
	fileops.write(_compute('AUC', feats, data, sk))

def _run_combined ():
	return None
	data=[_get_data_dna(), _get_data_dna(), _get_data_rand()]
	feats=[_get_feats_string(data[0]),
		_get_feats_string(data[1]), _get_feats_real(data[2])]

	k=CombinedKernel()
	k.append_kernel(FixedDegreeStringKernel(10, 3))
	k.append_kernel(LinearStringKernel(10))
	k.append_kernel(GaussianKernel(10, 5.5))

	train=CombinedFeatures()
	train.append_feature_obj(feats[0]['train'])
	train.append_feature_obj(feats[1]['train'])
	train.append_feature_obj(feats[2]['train'])
	k.init(train, train)

	test=CombinedFeatures()
	test.append_feature_obj(feats[0]['test'])
	test.append_feature_obj(feats[1]['test'])
	test.append_feature_obj(feats[2]['test'])
	k.init(train, test)

def _run_custom ():
	return None
	#fileops.write(_compute('Custom', feats, data))

#def _run_intfeats ():
#def _run_shortfeats ():
#def _run_ulongfeats ():
#	data=_get_data_rand(dattype=uint)
#	feats=_get_feats_simple('Int', data)
#
#	fileops.write(_compute('Int', feats, data))

def _run_distance ():
	data=_get_data_rand()
	feats=_get_feats_simple('Real', data)
	distance=RealDistance()

	fileops.write(_compute('Distance', feats, data, distance))

def _run_feats_byte ():
	data=_get_data_rand(dattype=ubyte)
	feats=_get_feats_simple('Byte', data)

	#fileops.write(_compute('Byte', feats, data))
	fileops.write(_compute('LinearByte', feats, data))

def _run_feats_char ():
	data=_get_data_rand(dattype=character)
	feats=_get_feats_simple('Char', data)

	#fileops.write(_compute('Char', feats, data))

def _run_mindygram ():
	data=_get_data_dna()
	feats={'train':MindyGramFeatures(STR_ALPHABET, 'freq', '%20.,', 0),
		'test':MindyGramFeatures(STR_ALPHABET, 'freq', '%20.,', 0)}

	fileops.write(_compute('MindyGram', feats, data, 'MEASURE', 1.5))

def _run_feats_real ():
	data=_get_data_rand()
	feats=_get_feats_simple('Real', data)

	fileops.write(_compute('Chi2', feats, data, 1.2, 10))
	fileops.write(_compute('Const', feats, data, 23.))
	fileops.write(_compute('Diag', feats, data, 23.))
	fileops.write(_compute('Gaussian', feats, data, 1.3))
	fileops.write(_compute('GaussianShift', feats, data, 1.3, 2, 1))
	fileops.write(_compute('Linear', feats, data, 1.))
	fileops.write(_compute('Poly', feats, data, 3, True, True))
	fileops.write(_compute('Poly', feats, data, 3, False, True))
	fileops.write(_compute('Poly', feats, data, 3, True, False))
	fileops.write(_compute('Poly', feats, data, 3, False, False))
	#fileops.write(_compute('Real', feats, data))
	fileops.write(_compute('Sigmoid', feats, data, 10, 1.1, 1.3))
	fileops.write(_compute('Sigmoid', feats, data, 10, 0.5, 0.7))

	fileops.write(_compute_svm('Gaussian', feats, data, .017, 1.5))
	fileops.write(_compute_svm('Gaussian', feats, data, .23, 1.5))

	feats=_get_feats_simple('Real', data, sparse=True)
	fileops.write(_compute('SparseGaussian', feats, data, 1.3))
	fileops.write(_compute('SparseLinear', feats, data, 1.))
	fileops.write(_compute('SparsePoly', feats, data, 10, 3, True, True))
	#fileops.write(_compute('SparseReal', feats, data))

def _run_feats_string ():
	data=_get_data_dna()
	feats=_get_feats_string('Char', data)

	fileops.write(_compute('FixedDegreeString', feats, data, 3))
	fileops.write(_compute('LinearString', feats, data))
	fileops.write(_compute('LocalAlignmentString', feats, data))
	fileops.write(_compute('PolyMatchString', feats, data, 3, True))
	fileops.write(_compute('PolyMatchString', feats, data, 3, False))
	fileops.write(_compute('SimpleLocalityImprovedString', feats, data, 5, 7, 5))
	#fileops.write(_compute('StringReal', feats, data))

	fileops.write(_compute('WeightedDegreeString', feats, data, 20, 0))
	fileops.write(_compute('WeightedDegreePositionString', feats, data, 20))

	# buggy:
	#fileops.write(_compute('LocalityImprovedString', feats, data, 51, 5, 7))

	#feats=_get_feats_string('Ulong', data)
	#fileops.write(_compute('CommUlongString', feats, data, False, FULL_NORMALIZATION))

def _run_feats_word ():
	data=_get_data_rand(dattype=ushort)
	feats=_get_feats_simple('Word', data)

	fileops.write(_compute('CanberraWord', feats, data, 1.7))
	fileops.write(_compute('HammingWord', feats, data, 1.3, False))
	fileops.write(_compute('LinearWord', feats, data))
	fileops.write(_compute('ManhattenWord', feats, data, 1.5))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, True))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, False))
	#fileops.write(_compute('Word', feats, data))
	fileops.write(_compute('WordMatch', feats, data, 3))

	feats=_get_feats_simple('Word', data, sparse=True)
	#fileops.write(_compute('SparseWord', feats, data))

def _run_feats_wordstring ():
	data=_get_data_dna()
	feats=_get_feats_wordstring(data)

	fileops.write(_compute('CommWordString', feats, data, False, FULL_NORMALIZATION))
	fileops.write(_compute('WeightedCommWordString', feats, data, False, FULL_NORMALIZATION))

def _run_pluginestimate ():
	pass

def run ():
	#_run_auc()
	_run_feats_byte()
	_run_feats_char()
	_run_combined()
	_run_custom()
	#_run_distance()
	#_run_mindygram()
	_run_pluginestimate()
	_run_feats_real()
	_run_feats_string()
	_run_feats_wordstring()
	_run_feats_word()
