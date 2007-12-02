from numpy import *
from numpy.random import *
from shogun.Features import *
from shogun.Kernel import *
from shogun.Library import FULL_NORMALIZATION
from shogun.Classifier import *
from shogun.Distance import *

import fileops
import featops
import dataops
from klist import KLIST

##################################################################
## subkernel funs
##################################################################

def _compute_subkernels (name, feats, kernel, output):
	kernel.init(feats['train'], feats['train'])
	output['km_train']=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	output['km_test']=kernel.get_kernel_matrix()
	output.update(fileops.get_output_params(name))

	return [name, output]

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

def _get_subkernel_output_params (subkernel, data, num):
	prefix='subkernel'+num+'_'
	output={}

	output[prefix+'name']=subkernel[0]
	#FIXME: size soon to be removed from constructor
	output[prefix+'kparam0_size']='10'
	output[prefix+'data_train']=matrix(data['train'])
	output[prefix+'data_test']=matrix(data['test'])
	output.update(fileops.get_output_params(
		subkernel[0], subkernel[1:], prefix, 1))

	return output

def _run_auc ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	width=1.5
	subkernels=[['Gaussian', width]]
	sk=GaussianKernel(feats['train'], feats['test'], width)
	output=_get_subkernel_output_params(subkernels[0], data, '0')

	data=dataops.get_rand(ushort, rows=2, max_train=dataops.LEN_TRAIN,
		max_test=dataops.LEN_TEST)
	feats=featops.get_simple('Word', data)
	#FIXME: size soon to be removed from constructor
	kernel=AUCKernel(10, sk)
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
		output.update(_get_subkernel_output_params(
			subkernels[i], data_sk, str(i)))

	fileops.write(_compute_subkernels('Combined', feats, kernel, output))

def _run_subkernels ():
	_run_auc()
	_run_combined()

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
	output.update(fileops.get_output_params(name, args))

	return [name, output]

def _compute_svm (name, feats, data, params, *args):
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
	output.update(fileops.get_output_params(name, args))

	return [fileops.SVM+name, output]

def _compute_pie (name, feats, data):
	pie=PluginEstimate()
	kfun=eval(name+'Kernel')

	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	pie.train(feats['train'], l, .1, -.1)
	k=kfun(feats['train'], feats['train'], pie)

	k.init(feats['train'], feats['test'])
	pie.set_testfeatures(feats['test'])
	pie.test()
	classified=pie.classify().get_labels()

	output=_get_output(name, {
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'labels':labels,
		'classified':classified
	})

	return [name, output]

##################################################################
## run funcs
##################################################################

def _run_custom ():
	dim_square=7
	name='Custom'
	data=dataops.get_rand(dim_square=dim_square)
	feats=featops.get_simple('Real', data)
	data=data['train']
	symdata=data+data.T

	lowertriangle=array([ symdata[(x,y)] for x in xrange(symdata.shape[1]) for y in xrange(symdata.shape[0]) if y<=x ])
	k=CustomKernel(feats['train'], feats['train'])
	k.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=k.get_kernel_matrix()
	k.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=k.get_kernel_matrix()
	k.set_full_kernel_matrix_from_full(data)
	km_fullfull=k.get_kernel_matrix()

	output={
		'km_triangletriangle':km_triangletriangle,
		'km_fulltriangle':km_fulltriangle,
		'km_fullfull':km_fullfull,
		'symdata':matrix(symdata),
		'data':matrix(data),
		'dim_square':dim_square
	}
	output.update(fileops.get_output_params(name))

	fileops.write([name, output])

def _run_distance ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	distance=CanberraMetric()
	fileops.write(_compute('Distance', feats, data, 1.7, distance))

def _run_feats_byte ():
	data=dataops.get_rand(type=ubyte)
	feats=featops.get_simple('Byte', data, RAWBYTE)

	fileops.write(_compute('LinearByte', feats, data))

def _run_mindygram ():
	data=dataops.get_dna()
	feats={'train':MindyGramFeatures('DNA', 'freq', '%20.,', 0),
		'test':MindyGramFeatures('DNA', 'freq', '%20.,', 0)}

	fileops.write(_compute('MindyGram', feats, data, 'MEASURE', 1.5))

def _run_feats_real ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)

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
	fileops.write(_compute('Sigmoid', feats, data, 10, 1.1, 1.3))
	fileops.write(_compute('Sigmoid', feats, data, 10, 0.5, 0.7))

	feats=featops.get_simple('Real', data, sparse=True)
	fileops.write(_compute('SparseGaussian', feats, data, 1.3))
	fileops.write(_compute('SparseLinear', feats, data, 1.))
	fileops.write(_compute('SparsePoly', feats, data, 10, 3, True, True))

def _run_feats_string ():
	data=dataops.get_dna()
	feats=featops.get_string('Char', data)

	fileops.write(_compute('FixedDegreeString', feats, data, 3))
	fileops.write(_compute('LinearString', feats, data))
	fileops.write(_compute('LocalAlignmentString', feats, data))
	fileops.write(_compute('PolyMatchString', feats, data, 3, True))
	fileops.write(_compute('PolyMatchString', feats, data, 3, False))
	fileops.write(_compute('SimpleLocalityImprovedString', feats, data, 5, 7, 5))

	fileops.write(_compute('WeightedDegreeString', feats, data, 20, 0))
	fileops.write(_compute('WeightedDegreePositionString', feats, data, 20))

	# buggy:
	#fileops.write(_compute('LocalityImprovedString', feats, data, 51, 5, 7))


def _run_feats_word ():
	#FIXME: greater max, lower variance?
	max=42
	data=dataops.get_rand(type=ushort, max_train=max, max_test=max)
	feats=featops.get_simple('Word', data)

	fileops.write(_compute('LinearWord', feats, data))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, True))
	fileops.write(_compute('PolyMatchWord', feats, data, 3, False))
	fileops.write(_compute('WordMatch', feats, data, 3))

def _run_feats_string_complex ():
	data=dataops.get_dna()
	feats=featops.get_string_complex('Word', data)

	fileops.write(_compute('CommWordString', feats, data, False, FULL_NORMALIZATION))
	fileops.write(_compute('WeightedCommWordString', feats, data, False, FULL_NORMALIZATION))

	feats=featops.get_string_complex('Ulong', data)
	fileops.write(_compute('CommUlongString', feats, data, False, FULL_NORMALIZATION))

def _run_pie ():
	data=dataops.get_rand(type=chararray)
	charfeats=featops.get_simple('Char', data)
	data=dataops.get_rand(type=ushort)
	feats=featops.get_simple('Word', data)
	feats['train'].obtain_from_char_features(charfeats['train'], 0, 1)
	feats['test'].obtain_from_char_features(charfeats['test'], 0, 1)

	fileops.write(_compute_pie('HistogramWord', feats, data))
	#fileops.write(_compute_pie('SalzbergWord', feats, data))

def _run_svm ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	width=1.5
	params_svm={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2, 'num_threads':1}

	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=.23
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=1.5
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['C']=30
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['epsilon']=1e-4
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['tube_epsilon']=1e-3
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))
	params_svm['num_threads']=16
	fileops.write(_compute_svm('Gaussian', feats, data, params_svm, width))

def run ():
	fileops.TYPE='Kernel'

	#_run_mindygram()
	#_run_pie()

	_run_custom()
	_run_distance()
	_run_subkernels()
	_run_svm()

	_run_feats_byte()
	_run_feats_real()
	_run_feats_string()
	_run_feats_string_complex()
	_run_feats_word()
