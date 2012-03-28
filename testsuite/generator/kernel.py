"""
Generator for Kernel

A word about args: it is organised as two correlated tuples, because the
order of the elements of dicts in Python is arbitrary, meaning that the item
added first might be the last when iterating over the dict.
"""

import numpy
import shogun.Kernel as kernel
from shogun.Features import CombinedFeatures, TOPFeatures, FKFeatures, CUBE, RAWBYTE
from shogun.Classifier import PluginEstimate
from shogun.Distance import CanberraMetric
from shogun.Distribution import HMM, LinearHMM, BW_NORMAL
from shogun.Library import Math_init_random

import fileop
import featop
import dataop
import category


##################################################################
## compute funcs
##################################################################

def _compute_pie (feats, params):
	"""Compute a kernel with PluginEstimate.

	@param feats kernel features
	@param params dict containing various kernel parameters
	"""

	output=fileop.get_output(category.KERNEL, params)

	lab, labels=dataop.get_labels(feats['train'].get_num_vectors())
	output['classifier_labels']=lab
	pie=PluginEstimate()
	pie.set_labels(labels)
	pie.set_features(feats['train'])
	pie.train()

	kfun=eval('kernel.'+params['name']+'Kernel')
	kern=kfun(feats['train'], feats['train'], pie)
	output['kernel_matrix_train']=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	pie.set_features(feats['test'])
	output['kernel_matrix_test']=kern.get_kernel_matrix()

	classified=pie.apply().get_labels()
	output['classifier_classified']=classified

	fileop.write(category.KERNEL, output)


def _compute_top_fisher (feats, pout):
	"""Compute PolyKernel with TOP or FKFeatures

	@param feats features of the kernel
	@param pout previously gathered data ready to be written to file
	"""

	params={
		'name': 'Poly',
		'accuracy': 1e-6,
		'args': {
			'key': ('size', 'degree', 'inhomogene'),
			'val': (10, 1, False)
		}
	}
	output=fileop.get_output(category.KERNEL, params)
	output.update(pout)

	kfun=eval('kernel.'+params['name']+'Kernel')
	kern=kfun(feats['train'], feats['train'], *params['args']['val'])
	output['kernel_matrix_train']=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	output['kernel_matrix_test']=kern.get_kernel_matrix()

	fileop.write(category.KERNEL, output)


def _compute (feats, params, pout=None):
	"""
	Compute a kernel and write gathered data to file.

	@param name name of the kernel
	@param feats features of the kernel
	@param params dict with parameters to kernel
	@param pout previously gathered data ready to be written to file
	"""

	output=fileop.get_output(category.KERNEL, params)
	if pout:
		output.update(pout)

	kfun=eval('kernel.'+params['name']+'Kernel')
	if params.has_key('args'):
		kern=kfun(*params['args']['val'])
	else:
		kern=kfun()

	if params.has_key('normalizer'):
		kern.set_normalizer(params['normalizer'])
	kern.init(feats['train'], feats['train'])

	output['kernel_matrix_train']=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	output['kernel_matrix_test']=kern.get_kernel_matrix()

	fileop.write(category.KERNEL, output)


##################################################################
## run funcs
##################################################################

def _run_auc ():
	"""Run AUC kernel."""

	# handle subkernel
	params={
		'name': 'Gaussian',
		'data': dataop.get_rand(),
		'feature_class': 'simple',
		'feature_type': 'Real',
		'args': {'key': ('size', 'width'), 'val': (10, 1.7)}
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	subk=kernel.GaussianKernel(*params['args']['val'])
	subk.init(feats['train'], feats['test'])
	output=fileop.get_output(category.KERNEL, params, 'subkernel0_')

	# handle AUC
	params={
		'name': 'AUC',
		'data': dataop.get_rand(numpy.ushort, num_feats=2,
			max_train=dataop.NUM_VEC_TRAIN, max_test=dataop.NUM_VEC_TEST),
		'feature_class': 'simple',
		'feature_type': 'Word',
		'accuracy': 1e-8,
		'args': {'key': ('size', 'subkernel'), 'val': (10, subk)}
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	_compute(feats, params, output)


def _run_combined ():
	"""Run Combined kernel."""

	kern=kernel.CombinedKernel()
	feats={'train': CombinedFeatures(), 'test': CombinedFeatures()}
	output={}
	params={
		'name': 'Combined',
		'accuracy': 1e-7
	}
	subkdata=[
		{
			'name': 'FixedDegreeString',
			'feature_class': 'string',
			'feature_type': 'Char',
			'args': {'key': ('size', 'degree'), 'val': (10, 3)}
		},
		{
			'name': 'PolyMatchString',
			'feature_class': 'string',
			'feature_type': 'Char',
			'args': {
				'key': ('size', 'degree', 'inhomogene'),
				'val': (10, 3, True)
			}
		},
		{
			'name': 'LocalAlignmentString',
			'feature_class': 'string',
			'feature_type': 'Char',
			'args': {'key': ('size',), 'val': (10,)}
		}
	]

	i=0
	for sd in subkdata:
		kfun=eval('kernel.'+sd['name']+'Kernel')
		subk=kfun(*sd['args']['val'])
		sd['data']=dataop.get_dna()
		subkfeats=featop.get_features(
			sd['feature_class'], sd['feature_type'], sd['data'])
		output.update(
			fileop.get_output(category.KERNEL, sd, 'subkernel'+str(i)+'_'))

		kern.append_kernel(subk)
		feats['train'].append_feature_obj(subkfeats['train'])
		feats['test'].append_feature_obj(subkfeats['test'])

		i+=1

	output.update(fileop.get_output(category.KERNEL, params))
	kern.init(feats['train'], feats['train'])
	output['kernel_matrix_train']=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	output['kernel_matrix_test']=kern.get_kernel_matrix()

	fileop.write(category.KERNEL, output)


def _run_subkernels ():
	"""Run all kernels handling subkernels."""

	_run_auc()
	_run_combined()


def _run_custom ():
	"""Run Custom kernel."""

	params={
		'name': 'Custom',
		'accuracy': 1e-7,
		'feature_class': 'simple',
		'feature_type': 'Real'
	}
	dim_square=7
	data=dataop.get_rand(dim_square=dim_square)
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], data)
	data=data['train']
	symdata=data+data.T

	lowertriangle=numpy.array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])
	kern=kernel.CustomKernel()
	#kern.init(feats['train'], feats['train']
	kern.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kern.get_kernel_matrix()
	kern.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kern.get_kernel_matrix()
	kern.set_full_kernel_matrix_from_full(data)
	km_fullfull=kern.get_kernel_matrix()

	output={
		'kernel_matrix_triangletriangle': km_triangletriangle,
		'kernel_matrix_fulltriangle': km_fulltriangle,
		'kernel_matrix_fullfull': km_fullfull,
		'kernel_symdata': numpy.matrix(symdata),
		'kernel_data': numpy.matrix(data),
		'kernel_dim_square': dim_square
	}
	output.update(fileop.get_output(category.KERNEL, params))

	fileop.write(category.KERNEL, output)


def _run_distance ():
	"""Run distance kernel."""

	params={
		'name': 'Distance',
		'accuracy': 1e-9,
		'feature_class': 'simple',
		'feature_type': 'Real',
		'data': dataop.get_rand(),
		'args': {
			'key': ('size', 'width', 'distance'),
			'val': (10, 1.7, CanberraMetric())
		}
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	_compute(feats, params)


#def _run_feats_byte ():
#	"""Run kernel with ByteFeatures."""
#
#	params={
#		'name': 'LinearByte',
#		'accuracy': 1e-8,
#		'feature_class': 'simple',
#		'feature_type': 'Byte',
#		'data': dataop.get_rand(dattype=numpy.ubyte),
#		'normalizer': kernel.AvgDiagKernelNormalizer()
#	}
#	feats=featop.get_features(params['feature_class'], params['feature_type'],
#		params['data'], RAWBYTE)
#
#	_compute(feats, params)


def _run_mindygram ():
	"""Run Mindygram kernel."""
	return

	params={
		'name': 'MindyGram',
		'accuracy': 1e-8,
		'data': dataop.get_dna(),
		'feature_class': 'mindy',
		'args': {'key': ('measure', 'width'), 'val': ('MEASURE', 1.5)}
	}
	feats={
		'train': MindyGramFeatures('DNA', 'freq', '%20.,', 0),
		'test': MindyGramFeatures('DNA', 'freq', '%20.,', 0)
	}

	_compute(feats, params)


def _run_feats_real ():
	"""Run kernel with RealFeatures."""

	params={
		'data': dataop.get_rand(),
		'accuracy': 1e-8,
		'feature_class': 'simple',
		'feature_type': 'Real'
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	sparsefeats=featop.get_features(
		params['feature_class'], params['feature_type'],
		params['data'], sparse=True)

	params['name']='Gaussian'
	params['args']={'key': ('size', 'width',), 'val': (10, 1.3)}
	_compute(feats, params)

	params['name']='GaussianShift'
	params['args']={
		'key': ('size', 'width', 'max_shift', 'shift_step'),
		'val': (10, 1.3, 2, 1)
	}
	_compute(feats, params)

    #params['name']='SparseGaussian'
	#params['args']={'key': ('size', 'width'), 'val': (10, 1.7)}
	#_compute(sparsefeats, params)

	params['accuracy']=0
	params['name']='Const'
	params['args']={'key': ('c',), 'val': (23.,)}
	_compute(feats, params)

	params['name']='Diag'
	params['args']={'key': ('size', 'diag'), 'val': (10, 23.)}
	_compute(feats, params)

	params['accuracy']=1e-9
	params['name']='Sigmoid'
	params['args']={
		'key': ('size', 'gamma', 'coef0'),
		'val': (10, 1.1, 1.3)
	}
	_compute(feats, params)
	params['args']['val']=(10, 0.5, 0.7)
	_compute(feats, params)

	params['name']='Chi2'
	params['args']={'key': ('size', 'width'), 'val': (10, 1.2)}
	_compute(feats, params)

    #params['accuracy']=1e-8
	#params['name']='SparsePoly'
	#params['args']={
	#	'key': ('size', 'degree', 'inhomogene'),
	#	'val': (10, 3, True)
	#}
    #_compute(sparsefeats, params)
	#params['args']['val']=(10, 3, False)
	#_compute(sparsefeats, params)

	params['name']='Poly'
	params['normalizer']=kernel.SqrtDiagKernelNormalizer()
	params['args']={
		'key': ('size', 'degree', 'inhomogene'),
		'val': (10, 3, True)
	}
	_compute(feats, params)
	params['args']['val']=(10, 3, False)
	_compute(feats, params)

    #params['normalizer']=kernel.AvgDiagKernelNormalizer()
	#del params['args']
	#params['name']='Linear'
	#_compute(feats, params)
	#params['name']='SparseLinear'
	#_compute(sparsefeats, params)


def _run_feats_string ():
	"""Run kernel with StringFeatures."""

	params = {
		'accuracy': 1e-9,
		'data': dataop.get_dna(),
		'feature_class': 'string',
		'feature_type': 'Char',
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	params['name']='FixedDegreeString'
	params['args']={'key': ('size', 'degree'), 'val': (10, 3)}
	_compute(feats, params)

	params['accuracy']=0
	params['name']='LocalAlignmentString'
	params['args']={'key': ('size',), 'val': (10,)}
	_compute(feats, params)

	params['accuracy']=1e-10
	params['name']='PolyMatchString'
	params['args']={
		'key': ('size', 'degree', 'inhomogene'),
		'val': (10, 3, True)
	}
	_compute(feats, params)
	params['args']['val']=(10, 3, False)
	_compute(feats, params)

	params['accuracy']=1e-15
	params['name']='SimpleLocalityImprovedString'
	params['args']={
		'key': ('size', 'length', 'inner_degree', 'outer_degree'),
		'val': (10, 5, 7, 5)
	}
	_compute(feats, params)
	# buggy:
	#params['name']='LocalityImprovedString'
	#_compute(feats, params)

	params['name']='WeightedDegreeString'
	params['accuracy']=1e-9
	params['args']={'key': ('degree',), 'val': (20,)}
	_compute(feats, params)
	params['args']={'key': ('degree',), 'val': (1,)}
	_compute(feats, params)

	params['name']='WeightedDegreePositionString'
	params['args']={'key': ('size', 'degree'), 'val': (10, 20)}
	_compute(feats, params)
	params['args']={'key': ('size', 'degree'), 'val': (10, 1)}
	_compute(feats, params)

	params['name']='OligoString'
	params['args']={'key': ('size', 'k', 'width'), 'val': (10, 3, 1.2)}
	_compute(feats, params)
	params['args']={'key': ('size', 'k', 'width'), 'val': (10, 4, 1.7)}
	_compute(feats, params)

	params['name']='LinearString'
	params['accuracy']=1e-8
	params['normalizer']=kernel.AvgDiagKernelNormalizer()
	del params['args']
	_compute(feats, params)


#def _run_feats_word ():
#	"""Run kernel with WordFeatures."""
#
#	maxval=42
#	params={
#		'name': 'LinearWord',
#		'accuracy': 1e-8,
#		'feature_class': 'simple',
#		'feature_type': 'Word',
#		'data': dataop.get_rand(
#			dattype=numpy.ushort, max_train=maxval, max_test=maxval),
#		'normalizer': kernel.AvgDiagKernelNormalizer()
#	}
#	feats=featop.get_features(
#		params['feature_class'], params['feature_type'], params['data'])
#
#	_compute(feats, params)


def _run_feats_string_complex ():
	"""Run kernel with complex StringFeatures."""

	params={
		'data': dataop.get_dna(),
		'feature_class': 'string_complex'
	}

	params['feature_type']='Word'
	wordfeats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	params['name']='CommWordString'
	params['accuracy']=1e-9
	params['args']={'key': ('size', 'use_sign'), 'val': (10, False)}
	_compute(wordfeats, params)
	params['name']='WeightedCommWordString'
	_compute(wordfeats, params)

	params['name']='PolyMatchWordString'
	params['accuracy']=1e-10
	params['args']={
		'key': ('size', 'degree', 'inhomogene'),
		'val': (10, 3, True)
	}
	_compute(wordfeats, params)
	params['args']['val']=(10, 3, False)
	_compute(wordfeats, params)

	params['name']='MatchWordString'
	params['args']={'key': ('size', 'degree'), 'val': (10, 3)}
	_compute(wordfeats, params)

	params['feature_type']='Ulong'
	params['accuracy']=1e-9
	ulongfeats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	params['name']='CommUlongString'
	params['args']={'key': ('size', 'use_sign'), 'val': (10, False)}
	_compute(ulongfeats, params)


def _run_pie ():
	"""Run kernel with PluginEstimate."""

	params={
		'data': dataop.get_dna(),
		'accuracy': 1e-6,
		'feature_class': 'string_complex',
		'feature_type': 'Word'
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	params['name']='HistogramWordString'
	_compute_pie(feats, params)
	params['name']='SalzbergWordString'
	_compute_pie(feats, params)


def _run_top_fisher ():
	"""Run Linear Kernel with {Top,Fisher}Features."""

	# put some constantness into randomness
	Math_init_random(dataop.INIT_RANDOM)

	data=dataop.get_cubes(4, 8)
	prefix='topfk_'
	params={
		prefix+'N': 3,
		prefix+'M': 6,
		prefix+'pseudo': 1e-1,
		prefix+'order': 1,
		prefix+'gap': 0,
		prefix+'reverse': False,
		prefix+'alphabet': 'CUBE',
		prefix+'feature_class': 'string_complex',
		prefix+'feature_type': 'Word',
		prefix+'data_train': numpy.matrix(data['train']),
		prefix+'data_test': numpy.matrix(data['test'])
	}

	wordfeats=featop.get_features(
		params[prefix+'feature_class'], params[prefix+'feature_type'],
		data, eval(params[prefix+'alphabet']),
		params[prefix+'order'], params[prefix+'gap'], params[prefix+'reverse'])
	pos_train=HMM(wordfeats['train'],
		params[prefix+'N'], params[prefix+'M'], params[prefix+'pseudo'])
	pos_train.train()
	pos_train.baum_welch_viterbi_train(BW_NORMAL)
	neg_train=HMM(wordfeats['train'],
		params[prefix+'N'], params[prefix+'M'], params[prefix+'pseudo'])
	neg_train.train()
	neg_train.baum_welch_viterbi_train(BW_NORMAL)
	pos_test=HMM(pos_train)
	pos_test.set_observations(wordfeats['test'])
	neg_test=HMM(neg_train)
	neg_test.set_observations(wordfeats['test'])
	feats={}

	feats['train']=TOPFeatures(10, pos_train, neg_train, False, False)
	feats['test']=TOPFeatures(10, pos_test, neg_test, False, False)
	params[prefix+'name']='TOP'
	_compute_top_fisher(feats, params)

	feats['train']=FKFeatures(10, pos_train, neg_train)
	feats['train'].set_opt_a(-1) #estimate prior
	feats['test']=FKFeatures(10, pos_test, neg_test)
	feats['test'].set_a(feats['train'].get_a()) #use prior from training data
	params[prefix+'name']='FK'
	_compute_top_fisher(feats, params)


def run ():
	"""Run generator for all kernels."""

	#_run_mindygram()
	_run_top_fisher()
	_run_pie()
	_run_custom()
	_run_distance()
	_run_subkernels()

    #_run_feats_byte()
	_run_feats_real()
	_run_feats_string()
	_run_feats_string_complex()
    #_run_feats_word()
