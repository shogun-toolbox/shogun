from shogun.Features import *
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Distance import *
from numpy import *

import util

########################################################################
# kernel computation
########################################################################

def _kernel (input, feats):
	kfun=eval(input['name']+'Kernel')
	args=util.get_args(input, 'kernel_arg')

	k=kfun(feats['train'], feats['train'], *args)
	train=max(abs(input['km_train']-k.get_kernel_matrix()).flat)
	k.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-k.get_kernel_matrix()).flat)

	return util.check_accuracy(input['accuracy'], train=train, test=test)

def _add_subkernels (subkernels):
	for idx, sk in enumerate(subkernels):
		fun=eval(sk['name']+'Kernel')
		args=util.get_args(sk, 'kernel_arg')
		subkernels[idx]['kernel']=fun(*args)
	return subkernels

def _get_subkernels (input):
	subkernels=len(input)*[None]
	len_subkernel=len('subkernel')

	for i in input:
		if i.find('subkernel')==-1:
			continue

		try:
			idx=int(i[len_subkernel])
		except ValueError:
			raise ValueError, 'Wrong input data subkernel: "%s"!' % i

		# get item's name
		item=i[i.find('_')+1:]

		# weird behaviour of python if subkernels is inited with {}, so
		# have to do this:
		if subkernels[idx] is None:
			subkernels[idx]={}

		subkernels[idx][item]=input[i]

	# weed out empty subkernels
	subkernels=filter(lambda x: x is not None, subkernels)
	return _add_subkernels(subkernels)

def _kernel_combined (input):
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}

	subkernels=_get_subkernels(input)
	for sk in subkernels:
		feats_sk=eval('util.get_feats_'+sk['feature_class']+'(sk)')
		kernel.append_kernel(sk['kernel'])
		feats['train'].append_feature_obj(feats_sk['train'])
		feats['test'].append_feature_obj(feats_sk['test'])

	return _kernel_subkernels(input, feats, kernel)

def _kernel_auc (input):
	sk=_get_subkernels(input)[0]
	feats_sk=eval('util.get_feats_'+sk['feature_class']+'(sk)')
	sk['kernel'].init(feats_sk['train'], feats_sk['test'])

	feats={
		'train':WordFeatures(input['data_train'].astype(eval(input['data_type']))),
		'test':WordFeatures(input['data_test'].astype(eval(input['data_type'])))}
	kernel=AUCKernel(10, sk['kernel'])

	return _kernel_subkernels(input, feats, kernel)

def _kernel_subkernels (input, feats, kernel):
	kernel.init(feats['train'], feats['train'])
	train=max(abs(input['km_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(input['accuracy'],
		train=train, test=test)

def _kernel_custom (input):
	feats={'train':RealFeatures(input['data']),
		'test':RealFeatures(input['data'])}

	symdata=input['symdata']

	lowertriangle = array([ symdata[(x,y)] for x in xrange(symdata.shape[1]) for y in xrange(symdata.shape[0]) if y<=x ])

	k=CustomKernel(feats['train'], feats['train'])
	k.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	triangletriangle=max(abs(input['km_triangletriangle']-k.get_kernel_matrix()).flat)
	k.set_triangle_kernel_matrix_from_full(input['symdata'])
	fulltriangle=max(abs(input['km_fulltriangle']-k.get_kernel_matrix()).flat)
	k.set_full_kernel_matrix_from_full(input['data'])
	fullfull=max(abs(input['km_fullfull']-k.get_kernel_matrix()).flat)

	return util.check_accuracy(input['accuracy'],
		triangletriangle=triangletriangle, fulltriangle=fulltriangle,
		fullfull=fullfull)

def _kernel_pie (input):
	print 'Not implemented yet!'
	return True

########################################################################
# public
########################################################################

def test (input):
	names=['Combined', 'AUC', 'Custom']
	for n in names:
		if input['name']==n:
			return eval('_kernel_'+n.lower()+'(input)')

	names=['HistogramWord', 'SalzbergWord']
	for n in names:
		if input['name']==n:
			return _kernel_pie(input)

	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)
	return _kernel(input, feats)

