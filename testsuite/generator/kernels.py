from numpy.random import *
from numpy import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import FULL_NORMALIZATION
from shogun.Classifier import *

import fileops
import featops
import dataops
from klist import KLIST

def _get_params_global (name):
	kdata=KLIST[name]
	params={}

	params['data_class']=kdata[0][0]
	params['data_type']=kdata[0][1]
	params['feature_class']=kdata[1][0]
	params['feature_type']=kdata[1][1]
	params['accuracy']=kdata[3]
	if kdata[1][0]=='string' or (kdata[1][0]=='simple' and kdata[1][1]=='char'):
		params['alphabet']='DNA'
		params['seqlen']=dataops.LEN_SEQ
	elif kdata[1][0]=='simple' and kdata[1][1]=='byte':
		params['alphabet']='RAWBYTE'
		params['seqlen']=dataops.LEN_SEQ
	elif kdata[1][0]=='string_complex':
		params['order']=featops.WORDSTRING_ORDER
		params['gap']=featops.WORDSTRING_GAP
		params['reverse']=featops.WORDSTRING_REVERSE
		params['alphabet']='DNA'
		params['seqlen']=dataops.LEN_SEQ
		params['feature_obtain']=kdata[1][2]

	return params

##################################################################
## compute/kernel funcs
##################################################################

def compute (name, feats, data, *args):
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
		# a bit awkward to have this specialised cond here:
		if pname.find('distance')!=-1:
			output[pname]=args[i].__class__.__name__
		else:
			output[pname]=args[i]

	return [name, output]

def compute_svm (name, feats, data, params, *args):
	kfun=eval(name+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	k.parallel.set_num_threads(params['num_threads'])

	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svm=SVMLight(params['C'], k, l)
	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])
	svm.set_tube_epsilon(params['tube_epsilon'])
	svm.train()
	alphas=svm.get_alphas()
	bias=svm.get_bias()
	support_vectors=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	classified=svm.classify().get_labels()

	output={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'C':params['C'],
		'epsilon':params['epsilon'],
		'tube_epsilon':params['tube_epsilon'],
		'num_threads':params['num_threads'],
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

def compute_subkernels (name, feats, kernel, output):
	kernel.init(feats['train'], feats['train'])
	output['km_train']=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	output['km_test']=kernel.get_kernel_matrix()
	output.update(_get_params_global(name))
	return [name, output]

##################################################################
## special cases
##################################################################

def _run_custom ():
	return None
	#fileops.write(compute('Custom', feats, data))

def _run_feats_byte ():
	data=dataops.get_rand(type=ubyte)
	feats=featops.get_simple('Byte', data, RAWBYTE)

	fileops.write(compute('LinearByte', feats, data))

def _run_mindygram ():
	data=dataops.get_dna()
	feats={'train':MindyGramFeatures('DNA', 'freq', '%20.,', 0),
		'test':MindyGramFeatures('DNA', 'freq', '%20.,', 0)}

	fileops.write(compute('MindyGram', feats, data, 'MEASURE', 1.5))

def _run_feats_real ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)

	fileops.write(compute('Chi2', feats, data, 1.2, 10))
	fileops.write(compute('Const', feats, data, 23.))
	fileops.write(compute('Diag', feats, data, 23.))
	fileops.write(compute('Gaussian', feats, data, 1.3))
	fileops.write(compute('GaussianShift', feats, data, 1.3, 2, 1))
	fileops.write(compute('Linear', feats, data, 1.))
	fileops.write(compute('Poly', feats, data, 3, True, True))
	fileops.write(compute('Poly', feats, data, 3, False, True))
	fileops.write(compute('Poly', feats, data, 3, True, False))
	fileops.write(compute('Poly', feats, data, 3, False, False))
	fileops.write(compute('Sigmoid', feats, data, 10, 1.1, 1.3))
	fileops.write(compute('Sigmoid', feats, data, 10, 0.5, 0.7))

	feats=featops.get_simple('Real', data, sparse=True)
	fileops.write(compute('SparseGaussian', feats, data, 1.3))
	fileops.write(compute('SparseLinear', feats, data, 1.))
	fileops.write(compute('SparsePoly', feats, data, 10, 3, True, True))

def _run_feats_string ():
	data=dataops.get_dna()
	feats=featops.get_string('Char', data)

	fileops.write(compute('FixedDegreeString', feats, data, 3))
	fileops.write(compute('LinearString', feats, data))
	fileops.write(compute('LocalAlignmentString', feats, data))
	fileops.write(compute('PolyMatchString', feats, data, 3, True))
	fileops.write(compute('PolyMatchString', feats, data, 3, False))
	fileops.write(compute('SimpleLocalityImprovedString', feats, data, 5, 7, 5))

	fileops.write(compute('WeightedDegreeString', feats, data, 20, 0))
	fileops.write(compute('WeightedDegreePositionString', feats, data, 20))

	# buggy:
	#fileops.write(compute('LocalityImprovedString', feats, data, 51, 5, 7))


def _run_feats_word ():
	#FIXME: greater max, lower variance?
	#max=2**16-1
	max=42
	data=dataops.get_rand(type=ushort, max_train=max, max_test=max)
	feats=featops.get_simple('Word', data)

	fileops.write(compute('CanberraWord', feats, data, 1.7))
	fileops.write(compute('HammingWord', feats, data, 1.3, False))
	fileops.write(compute('LinearWord', feats, data))
	fileops.write(compute('ManhattanWord', feats, data, 1.5))
	fileops.write(compute('PolyMatchWord', feats, data, 3, True))
	fileops.write(compute('PolyMatchWord', feats, data, 3, False))
	fileops.write(compute('WordMatch', feats, data, 3))

def _run_feats_string_complex ():
	data=dataops.get_dna()
	feats=featops.get_string_complex('Word', data)

	fileops.write(compute('CommWordString', feats, data, False, FULL_NORMALIZATION))
	fileops.write(compute('WeightedCommWordString', feats, data, False, FULL_NORMALIZATION))

	feats=featops.get_string_complex('Ulong', data)
	fileops.write(compute('CommUlongString', feats, data, False, FULL_NORMALIZATION))

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
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	width=1.5
	subkernels=[['Gaussian', width]]
	sk=GaussianKernel(feats['train'], feats['test'], width)
	output=_get_subkernel_params(subkernels[0], data, '0')

	data=dataops.get_rand(ushort, 2, dataops.LEN_TRAIN, dataops.LEN_TEST)
	feats=featops.get_simple('Word', data)
	#FIXME: size soon to be removed from constructor
	kernel=AUCKernel(10, sk)
	output['data_train']=matrix(data['train'])
	output['data_test']=matrix(data['test'])

	fileops.write(compute_subkernels('AUC', feats, kernel, output))

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
		data_sk=eval('dataops.get_'+kdata[0][0]+'('+kdata[0][1]+')')
		feats_sk=eval('featops.get_'+kdata[1][0]+"('"+kdata[1][1]+"', data_sk)")
		feats['train'].append_feature_obj(feats_sk['train'])
		feats['test'].append_feature_obj(feats_sk['test'])
		output.update(_get_subkernel_params(subkernels[i], data_sk, str(i)))

	fileops.write(compute_subkernels('Combined', feats, kernel, output))

def _run_subkernels ():
	_run_auc()
	_run_combined()


def _run_svm ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	width=1.5
	params_svm={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2, 'num_threads':1}

	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=.23
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=1.5
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=30
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['epsilon']=1e-4
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['tube_epsilon']=1e-3
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['num_threads']=16
	fileops.write(compute_svm('Gaussian', feats, data, params_svm, width))


def run ():
	#_run_custom()
	#_run_mindygram()
	#_run_pluginestimate()

	#_run_subkernels()
	#_run_svm()

	#_run_feats_byte()
	#_run_feats_real()
	#_run_feats_string()
	_run_feats_string_complex()
	#_run_feats_word()

