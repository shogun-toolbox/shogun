"""
Generator for Kernel
"""

import numpy
import shogun.Library as library
import shogun.Kernel as kernel
from shogun.Features import CombinedFeatures, TOPFeatures, FKFeatures
from shogun.Classifier import PluginEstimate
from shogun.Distance import CanberraMetric
from shogun.Distribution import HMM, Model, LinearHMM

import fileop
import featop
import dataop
import config

##################################################################
## subkernel funs
##################################################################

def _compute_subkernels (name, feats, kern, outdata):
	"""Compute for kernel handling subkernels.

	@param name Name of the kernel
	@param feats Features of the kernel
	@param kernel Instantiated kernel
	@param outdata Already gathered data for output into testcase file
	"""

	outdata['name']=name
	kern.init(feats['train'], feats['train'])
	outdata['km_train']=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	outdata['km_test']=kern.get_kernel_matrix()
	outdata.update(fileop.get_outdata(name, config.C_KERNEL))

	fileop.write(config.C_KERNEL, outdata)

def _get_subkernel_args (subkernel):
	"""Return argument list for a subkernel.

	@param subkernel Tuple containing data relevant to a subkernel
	"""

	args=''
	i=0
	while 1:
		try:
			args+=', '+(str(subkernel[1+i]))
			i+=1
		except IndexError:
			break

	return args

def _get_subkernel_outdata (subkernel, data, num):
	"""Return data to be written into the testcase's file for a subkernel

	@param subkernel Tuple containing data relevant to a subkernel
	@param data Train and test data
	@param num Index number of the subkernel for identification in file
	"""

	prefix='subkernel'+num+'_'
	outdata={}

	outdata[prefix+'name']=subkernel[0]
	#FIXME: size soon to be removed from constructor
	outdata[prefix+'kernel_arg0_size']='10'
	outdata[prefix+'data_train']=numpy.matrix(data['train'])
	outdata[prefix+'data_test']=numpy.matrix(data['test'])
	outdata.update(fileop.get_outdata(
		subkernel[0], config.C_KERNEL, subkernel[1:], prefix, 1))

	return outdata

def _run_auc ():
	"""Run AUC kernel."""

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	width=1.5
	subkernels=[['Gaussian', width]]
	subk=kernel.GaussianKernel(feats['train'], feats['test'], width)
	outdata=_get_subkernel_outdata(subkernels[0], data, '0')

	data=dataop.get_rand(numpy.ushort, num_feats=2,
		max_train=dataop.NUM_VEC_TRAIN, max_test=dataop.NUM_VEC_TEST)
	feats=featop.get_simple('Word', data)
	#FIXME: size soon to be removed from constructor
	kern=kernel.AUCKernel(10, subk)
	outdata['data_train']=numpy.matrix(data['train'])
	outdata['data_test']=numpy.matrix(data['test'])

	_compute_subkernels('AUC', feats, kern, outdata)

def _run_combined ():
	"""Run Combined kernel."""

	kern=kernel.CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}
	subkernels=[
		['FixedDegreeString', 3],
		['PolyMatchString', 3, True],
		['LinearString'],
#		['Gaussian', 1.7],
	]
	outdata={}

	for i in range(0, len(subkernels)):
		kdata=config.KERNEL[subkernels[i][0]]
		args=_get_subkernel_args(subkernels[i])
		#FIXME: size soon to be removed from constructor
		subk=eval('kernel.'+subkernels[i][0]+'Kernel(10'+args+')')
		kern.append_kernel(subk)
		data_subk=eval('dataop.get_'+kdata[0][0]+'('+kdata[0][1]+')')
		feats_subk=eval(
			'featop.get_'+kdata[1][0]+"('"+kdata[1][1]+"', data_subk)")
		feats['train'].append_feature_obj(feats_subk['train'])
		feats['test'].append_feature_obj(feats_subk['test'])
		outdata.update(_get_subkernel_outdata(
			subkernels[i], data_subk, str(i)))

	_compute_subkernels('Combined', feats, kern, outdata)

def _run_subkernels ():
	"""Run all kernels handling subkernels."""

	_run_auc()
	_run_combined()

##################################################################
## compute/kernel funcs
##################################################################

def _compute (name, feats, data, *args):
	"""Compute a kernel and gather data.

	@param name Name of the kernel
	@param feats Features of the kernel
	@param data Train and test data
	@param *args variable argument list for kernel's constructor
	"""

	fun=eval('kernel.'+name+'Kernel')
	kern=fun(feats['train'], feats['train'], *args)
	kern.init(feats['train'], feats['train'])
	km_train=kern.get_kernel_matrix()
	kern.init(feats['train'], feats['test'])
	km_test=kern.get_kernel_matrix()

	outdata={
		'name':name,
		'name_features':
			feats['train'].__class__.__name__.replace('Features', ''),
		'km_train':km_train,
		'km_test':km_test,
		'data_train':numpy.matrix(data['train']),
		'data_test':numpy.matrix(data['test'])
	}
	outdata.update(fileop.get_outdata(name, config.C_KERNEL, args))

	fileop.write(config.C_KERNEL, outdata)

def _compute_pie (name, feats, data):
	"""Compute a kernel with PluginEstimate.

	@param name Name of the kernel
	@param feats Features of the kernel
	@param data Train and test data
	"""

	pie=PluginEstimate()
	fun=eval('kernel.'+name+'Kernel')

	lab, labels=dataop.get_labels(feats['train'].get_num_vectors())
	pie.train(feats['train'], labels)
	kern=fun(feats['train'], feats['train'], pie)
	km_train=kern.get_kernel_matrix()

	kern.init(feats['train'], feats['test'])
	pie.set_testfeatures(feats['test'])
	pie.test()
	km_test=kern.get_kernel_matrix()
	classified=pie.classify().get_labels()

	outdata={
		'name':name,
		'km_train':km_train,
		'km_test':km_test,
		'data_train':numpy.matrix(data['train']),
		'data_test':numpy.matrix(data['test']),
		'classifier_labels':lab,
		'classifier_classified':classified
	}
	outdata.update(fileop.get_outdata(name, config.C_KERNEL))

	fileop.write(config.C_KERNEL, outdata)

##################################################################
## run funcs
##################################################################

def _run_custom ():
	"""Run Custom kernel."""

	dim_square=7
	name='Custom'
	data=dataop.get_rand(dim_square=dim_square)
	feats=featop.get_simple('Real', data)
	data=data['train']
	symdata=data+data.T

	lowertriangle=numpy.array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])
	kern=kernel.CustomKernel(feats['train'], feats['train'])
	kern.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kern.get_kernel_matrix()
	kern.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kern.get_kernel_matrix()
	kern.set_full_kernel_matrix_from_full(data)
	km_fullfull=kern.get_kernel_matrix()

	outdata={
		'name':name,
		'km_triangletriangle':km_triangletriangle,
		'km_fulltriangle':km_fulltriangle,
		'km_fullfull':km_fullfull,
		'symdata':numpy.matrix(symdata),
		'data':numpy.matrix(data),
		'dim_square':dim_square
	}
	outdata.update(fileop.get_outdata(name, config.C_KERNEL))

	fileop.write(config.C_KERNEL, outdata)

def _run_distance ():
	"""Run distance kernel."""

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	distance=CanberraMetric()
	_compute('Distance', feats, data, 1.7, distance)

def _run_feats_byte ():
	"""Run kernel with ByteFeatures."""

	data=dataop.get_rand(dattype=numpy.ubyte)
	feats=featop.get_simple('Byte', data, library.RAWBYTE)

	_compute('LinearByte', feats, data)

def _run_mindygram ():
	"""Run Mindygram kernel."""
	return

	data=dataop.get_dna()
	feats={'train':MindyGramFeatures('DNA', 'freq', '%20.,', 0),
		'test':MindyGramFeatures('DNA', 'freq', '%20.,', 0)}

	_compute('MindyGram', feats, data, 'MEASURE', 1.5)

def _run_feats_real ():
	"""Run kernel with RealFeatures."""

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)

	_compute('Chi2', feats, data, 1.2, 10)
	_compute('Const', feats, data, 23.)
	_compute('Diag', feats, data, 23.)
	_compute('Gaussian', feats, data, 1.3)
	_compute('GaussianShift', feats, data, 1.3, 2, 1)
	_compute('Linear', feats, data, 1.)
	_compute('Poly', feats, data, 3, True, True)
	_compute('Poly', feats, data, 3, False, True)
	_compute('Poly', feats, data, 3, True, False)
	_compute('Poly', feats, data, 3, False, False)
	_compute('Sigmoid', feats, data, 10, 1.1, 1.3)
	_compute('Sigmoid', feats, data, 10, 0.5, 0.7)

	feats=featop.get_simple('Real', data, sparse=True)
	_compute('SparseGaussian', feats, data, 1.3)
	_compute('SparseLinear', feats, data, 1.)
	_compute('SparsePoly', feats, data, 10, 3, True, True)

def _run_feats_string ():
	"""Run kernel with StringFeatures."""

	data=dataop.get_dna()
	feats=featop.get_string('Char', data)

	_compute('FixedDegreeString', feats, data, 3)
	_compute('LinearString', feats, data)
	_compute('LocalAlignmentString', feats, data)
	_compute('PolyMatchString', feats, data, 3, True)
	_compute('PolyMatchString', feats, data, 3, False)
	_compute('SimpleLocalityImprovedString', feats, data, 5, 7, 5)

	_compute('WeightedDegreeString', feats, data, 20)
	_compute('WeightedDegreePositionString', feats, data, 20)

	# buggy:
	#_compute('LocalityImprovedString', feats, data, 51, 5, 7)


def _run_feats_word ():
	"""Run kernel with WordFeatures."""

	maxval=42
	data=dataop.get_rand(dattype=numpy.ushort,
		max_train=maxval, max_test=maxval)
	feats=featop.get_simple('Word', data)

	_compute('LinearWord', feats, data)
	_compute('PolyMatchWord', feats, data, 3, True)
	_compute('PolyMatchWord', feats, data, 3, False)
	_compute('WordMatch', feats, data, 3)

def _run_feats_string_complex ():
	"""Run kernel with complex StringFeatures."""

	data=dataop.get_dna()
	feats=featop.get_string_complex('Word', data)

	_compute('CommWordString', feats, data, False, library.FULL_NORMALIZATION)
	_compute('WeightedCommWordString', feats, data, False,
		library.FULL_NORMALIZATION)

	feats=featop.get_string_complex('Ulong', data)
	_compute('CommUlongString', feats, data, False, library.FULL_NORMALIZATION)

def _run_pie ():
	"""Run kernel with PluginEstimate."""

	data=dataop.get_dna()
	feats=featop.get_string_complex('Word', data)

	_compute_pie('HistogramWord', feats, data)
	_compute_pie('SalzbergWord', feats, data)

def _run_top_fisher ():
	"""Run Linear Kernel with {Top,Fisher}Features."""

	N=3
	M=6
	pseudo=1e-10
	order=1
	data=dataop.get_cubes(2)

	# pointer/reference handling is a bit fucked up, so we can't reuse
	# variable names like feats, pos, neg
	feats={}
	pos={}
	neg={}
	feat=featop.get_string_complex('Word', data, library.CUBE, order)

	pos['train']=HMM(feat['train'], N, M, pseudo)
	neg['train']=HMM(feat['train'], N, M, pseudo)
	pos['test']=HMM(feat['test'], N, M, pseudo)
	neg['test']=HMM(feat['test'], N, M, pseudo)

	feats['train']=TOPFeatures(10, pos['train'], neg['train'],
		False, False)
	feats['test']=TOPFeatures(10, pos['test'], neg['test'],
		False, False)
	_compute('Linear', feats, data, 1.)

	feats['train']=FKFeatures(10, pos['train'], neg['train'])
	feats['test']=FKFeatures(10, pos['test'], neg['test'])
	_compute('Linear', feats, data, 1.)

def run ():
	"""Run generator for all kernels."""

	#_run_mindygram()
	_run_top_fisher()
	_run_pie()
	_run_custom()
	_run_distance()
	_run_subkernels()

	_run_feats_byte()
	_run_feats_real()
	_run_feats_string()
	_run_feats_string_complex()
	_run_feats_word()
