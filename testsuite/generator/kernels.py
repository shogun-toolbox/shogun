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
from klist import KLIST

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60
STR_ALPHABET='DNA'
SIZE_CACHE=10
WORDSTRING_ORDER=3
WORDSTRING_GAP=0
WORDSTRING_REVERSE=False

def _func_name():
	return _getframe(1).f_code.co_name

def _get_params_global (name):
	kdata=KLIST[name]
	params={}

	params['data_class']=kdata[0][0]
	params['data_type']=kdata[0][1]
	params['feature_class']=kdata[1][0]
	params['feature_type']=kdata[1][1]
	params['accuracy']=kdata[3]
	if kdata[1][0]=='string' or (
		kdata[1][0]=='simple' and (kdata[1][1]=='byte' or kdata[1][1]=='char')):
		params['alphabet']=STR_ALPHABET
		params['len_seq']=LEN_SEQ
	elif kdata[1][0]=='wordstring':
		params['order']=WORDSTRING_ORDER
		params['gap']=WORDSTRING_GAP
		params['reverse']=WORDSTRING_REVERSE
		params['alphabet']=STR_ALPHABET
		params['len_seq']=LEN_SEQ

	return params

##################################################################
## compute/kernel funcs
##################################################################

def _compute (name, feats, data, *args):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	km_train=k.get_kernel_matrix()
	k.init(feats['train'], feats['test'])
	km_test=k.get_kernel_matrix()

	output={
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	output.update(_get_params_global(name))

	for i in range(0, len(args)):
		pname='kparam'+str(i)+'_'+KLIST[name][2][i]
		output[pname]=args[i]

	return [name, output]

def _compute_svm (name, feats, data, C, num_threads, *args):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	k.parallel.set_num_threads(num_threads)

	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svm=SVMLight(C, k, l)
	svm.parallel.set_num_threads(num_threads)
	svm.train()
	alphas=svm.get_alphas()
	bias=svm.get_bias()
	support_vectors=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	classified=svm.classify().get_labels()

	output={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'C':C,
		'num_threads':num_threads,
		'alphas':alphas,
		'labels':labels,
		'bias':bias,
		'support_vectors':support_vectors,
		'classified':classified
	}
	output.update(_get_params_global(name))
	for i in range(0, len(args)):
		pname='kparam'+str(i)+'_'+KLIST[name][2][i]
		output[pname]=args[i]

	return [fileops.SVM+name, output]

def _compute_subkernels (name, feats, kernel, output):
	kernel.init(feats['train'], feats['train'])
	output['km_train']=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	output['km_test']=kernel.get_kernel_matrix()
	output.update(_get_params_global(name))

	return [name, output]

##################################################################
## feats and data funcs
##################################################################

def _get_data_rand (type=double, rows=ROWS):
	if type==double:
		return {'train':rand(rows, LEN_TRAIN), 'test':rand(rows, LEN_TEST)}
	else:
		# randint does not understand arg dtype
		train=randint(0, 42, (rows, LEN_TRAIN))
		test=randint(0, 42, (rows, LEN_TEST))
		return {'train':train.astype(type), 'test':test.astype(type)}

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
	type=type.capitalize()
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
	type=type.capitalize()
	train=eval('String'+type+"Features(alphabet)")
	train.set_string_features(data['train'])
	test=eval('String'+type+"Features(alphabet)")
	test.set_string_features(data['test'])

	return {'train':train, 'test':test}

def _get_feats_wordstring (data, alphabet=DNA, order=WORDSTRING_ORDER,
	gap=WORDSTRING_GAP, reverse=WORDSTRING_REVERSE):

	feats={}

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['train'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat, WORDSTRING_ORDER-1,
		WORDSTRING_ORDER, WORDSTRING_GAP, WORDSTRING_REVERSE)
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['test'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat, WORDSTRING_ORDER-1,
		WORDSTRING_ORDER, WORDSTRING_GAP, WORDSTRING_REVERSE)
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return feats


##################################################################
## special cases
##################################################################

def _run_custom ():
	return None
	#fileops.write(_compute('Custom', feats, data))

#def _run_intfeats ():
#def _run_shortfeats ():
#def _run_ulongfeats ():
#	data=_get_data_rand(type=uint)
#	feats=_get_feats_simple('Int', data)
#
#	fileops.write(_compute('Int', feats, data))

def _run_distance ():
	data=_get_data_rand()
	feats=_get_feats_simple('Real', data)
	distance=RealDistance()

	fileops.write(_compute('Distance', feats, data, distance))

def _run_feats_byte ():
	data=_get_data_rand(type=ubyte)
	feats=_get_feats_simple('Byte', data)

#	fileops.write(_compute('Byte', feats, data))
	fileops.write(_compute('LinearByte', feats, data))

def _run_feats_char ():
	data=_get_data_rand(type=character)
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
#	fileops.write(_compute('Real', feats, data))
	fileops.write(_compute('Sigmoid', feats, data, 10, 1.1, 1.3))
	fileops.write(_compute('Sigmoid', feats, data, 10, 0.5, 0.7))

	fileops.write(_compute_svm('Gaussian', feats, data, .017, 1, 1.5))
	fileops.write(_compute_svm('Gaussian', feats, data, .017, 16, 1.5))
	fileops.write(_compute_svm('Gaussian', feats, data, .23, 1, 1.5))
	fileops.write(_compute_svm('Gaussian', feats, data, 1.5, 1, 1.5))
	fileops.write(_compute_svm('Gaussian', feats, data, 30, 1, 1.5))

	feats=_get_feats_simple('Real', data, sparse=True)
	fileops.write(_compute('SparseGaussian', feats, data, 1.3))
	fileops.write(_compute('SparseLinear', feats, data, 1.))
	fileops.write(_compute('SparsePoly', feats, data, 10, 3, True, True))
#	fileops.write(_compute('SparseReal', feats, data))

def _run_feats_string ():
	data=_get_data_dna()
	feats=_get_feats_string('Char', data)

	fileops.write(_compute('FixedDegreeString', feats, data, 3))
	fileops.write(_compute('LinearString', feats, data))
	fileops.write(_compute('LocalAlignmentString', feats, data))
	fileops.write(_compute('PolyMatchString', feats, data, 3, True))
	fileops.write(_compute('PolyMatchString', feats, data, 3, False))
	fileops.write(_compute('SimpleLocalityImprovedString', feats, data, 5, 7, 5))
#	fileops.write(_compute('StringReal', feats, data))

	fileops.write(_compute('WeightedDegreeString', feats, data, 20, 0))
	fileops.write(_compute('WeightedDegreePositionString', feats, data, 20))

	# buggy:
	#fileops.write(_compute('LocalityImprovedString', feats, data, 51, 5, 7))

#	feats=_get_feats_string('Ulong', data)
#	fileops.write(_compute('CommUlongString', feats, data, False, FULL_NORMALIZATION))

def _run_feats_word ():
	data=_get_data_rand(type=ushort)
	feats=_get_feats_simple('Word', data)

	fileops.write(_compute('CanberraWord', feats, data, 1.7))
	fileops.write(_compute('HammingWord', feats, data, 1.3, False))
	fileops.write(_compute('LinearWord', feats, data))
	fileops.write(_compute('ManhattenWord', feats, data, 1.5))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, True))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, False))
#	fileops.write(_compute('Word', feats, data))
	fileops.write(_compute('WordMatch', feats, data, 3))

#	feats=_get_feats_simple('Word', data, sparse=True)
#	fileops.write(_compute('SparseWord', feats, data))

def _run_feats_wordstring ():
	data=_get_data_dna()
	feats=_get_feats_wordstring(data)

	fileops.write(_compute('CommWordString', feats, data, False, FULL_NORMALIZATION))
	fileops.write(_compute('WeightedCommWordString', feats, data, False, FULL_NORMALIZATION))

def _run_pluginestimate ():
	pass

def _get_subkernel_args (subkernel):
	args=''
	i=0
	while 1:
		try:
			args+=', '+(str(subkernel[1+i]))
			i+=1
		except IndexError:
			break

	return args

def _get_subkernel_params (subkernel, data, num):
	kdata=KLIST[subkernel[0]]
	params={}

	params['subkernel'+num+'_name']=subkernel[0]
	#FIXME: size soon to be removed from constructor
	params['subkernel'+num+'_kparam0_size']='10'
	params['subkernel'+num+'_feature_class']=kdata[1][0]
	params['subkernel'+num+'_feature_type']=kdata[1][1]
	params['subkernel'+num+'_data_train']=matrix(data['train'])
	params['subkernel'+num+'_data_test']=matrix(data['test'])
	params['subkernel'+num+'_data_class']=kdata[0][0]
	params['subkernel'+num+'_data_type']=kdata[0][1]

	i=0
	while 1:
		try:
			name='subkernel'+num+'_kparam'+str(i+1)+'_'+kdata[2][i]
			params[name]=subkernel[1+i]
			i+=1
		except IndexError:
			break

	return params

def _run_auc ():
	data=_get_data_rand()
	feats=_get_feats_simple('Real', data)
	width=1.5
	subkernels=[['Gaussian', width]]
	sk=GaussianKernel(feats['train'], feats['test'], width)

	data=_get_data_rand(type=ushort, rows=2)
	feats=_get_feats_simple('Word', data)
	#FIXME: size soon to be removed from constructor
	kernel=AUCKernel(10, sk)
	output=_get_subkernel_params(subkernels[0], data, '0')
	output['data_train']=matrix(data['train'])
	output['data_test']=matrix(data['test'])

	fileops.write(_compute_subkernels('AUC', feats, kernel, output))

def _run_combined ():
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}
	subkernels=[
		['FixedDegreeString', 3],
		['PolyMatchString', 3, True],
		['LinearString'],
#		['Gaussian', 1.7],
#		['CanberraWord', 1.7],
	]
	output={}

	for i in range(0, len(subkernels)):
		str_i=str(i)
		kdata=KLIST[subkernels[i][0]]
		args=_get_subkernel_args(subkernels[i])
		#FIXME: size soon to be removed from constructor
		sk=eval(subkernels[i][0]+'Kernel(10'+args+')')
		kernel.append_kernel(sk)
		data_sk=eval('_get_data_'+kdata[0][0]+'('+kdata[0][1]+')')
		feats_sk=eval('_get_feats_'+kdata[1][0]+"('"+kdata[1][1]+"', data_sk)")
		feats['train'].append_feature_obj(feats_sk['train'])
		feats['test'].append_feature_obj(feats_sk['test'])
		output.update(_get_subkernel_params(subkernels[i], data_sk, str(i)))

	fileops.write(_compute_subkernels('Combined', feats, kernel, output))

def _run_subkernels ():
	_run_auc()
	_run_combined()


def run ():
	#_run_custom()
	#_run_distance()
	#_run_mindygram()
	#_run_pluginestimate()
	_run_subkernels()

	#_run_feats_byte()
	#_run_feats_char()
	#_run_feats_real()
	#_run_feats_string()
	#_run_feats_word()
	#_run_feats_wordstring()
