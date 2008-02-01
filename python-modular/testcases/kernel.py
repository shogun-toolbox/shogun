"""
Test Kernel
"""

from shogun.Features import *
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Distance import *
from shogun.Classifier import PluginEstimate
from shogun.Distribution import HMM, BW_NORMAL
from numpy import array, ushort, ubyte, double

import util

########################################################################
# kernel computation
########################################################################

def _kernel (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	fun=eval(indata['name']+'Kernel')
	args=util.get_args(indata, 'kernel_arg')

	kernel=fun(feats['train'], feats['train'], *args)
	km_train=max(abs(indata['km_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(indata['km_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata['accuracy'], km_train=km_train, km_test=km_test)

def _add_subkernels (subkernels):
	for idx, subk in enumerate(subkernels):
		fun=eval(subk['name']+'Kernel')
		args=util.get_args(subk, 'kernel_arg')
		subkernels[idx]['kernel']=fun(*args)
	return subkernels

def _get_subkernels (indata):
	subkernels=len(indata)*[None]
	len_subkernel=len('subkernel')

	for i in indata:
		if i.find('subkernel')==-1:
			continue

		try:
			idx=int(i[len_subkernel])
		except ValueError:
			raise ValueError, 'Wrong indata data subkernel: "%s"!' % i

		# get item's name
		item=i[i.find('_')+1:]

		# weird behaviour of python if subkernels is inited with {}, so
		# have to do this:
		if subkernels[idx] is None:
			subkernels[idx]={}

		subkernels[idx][item]=indata[i]

	# weed out empty subkernels
	subkernels=filter(lambda x: x is not None, subkernels)
	return _add_subkernels(subkernels)

def _kernel_combined (indata):
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}

	subkernels=_get_subkernels(indata)
	for subk in subkernels:
		feats_subk=eval('util.get_feats_'+subk['feature_class']+'(subk)')
		kernel.append_kernel(subk['kernel'])
		feats['train'].append_feature_obj(feats_subk['train'])
		feats['test'].append_feature_obj(feats_subk['test'])

	return _kernel_subkernels(indata, feats, kernel)

def _kernel_auc (indata):
	subk=_get_subkernels(indata)[0]
	feats_subk=eval('util.get_feats_'+subk['feature_class']+'(subk)')
	subk['kernel'].init(feats_subk['train'], feats_subk['test'])

	feats={
		'train':WordFeatures(indata['data_train'].astype(eval(indata['data_type']))),
		'test':WordFeatures(indata['data_test'].astype(eval(indata['data_type'])))}
	kernel=AUCKernel(10, subk['kernel'])

	return _kernel_subkernels(indata, feats, kernel)

def _kernel_subkernels (indata, feats, kernel):
	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(indata['km_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(indata['km_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata['accuracy'],
		km_train=km_train, km_test=km_test)

def _kernel_custom (indata):
	feats={'train':RealFeatures(indata['data']),
		'test':RealFeatures(indata['data'])}

	symdata=indata['symdata']
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])
	kernel=CustomKernel(feats['train'], feats['train'])
	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	triangletriangle=max(abs(
		indata['km_triangletriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_triangle_kernel_matrix_from_full(indata['symdata'])
	fulltriangle=max(abs(
		indata['km_fulltriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_full_kernel_matrix_from_full(indata['data'])
	fullfull=max(abs(indata['km_fullfull']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata['accuracy'],
		triangletriangle=triangletriangle, fulltriangle=fulltriangle,
		fullfull=fullfull)

def _kernel_pie (indata):
	pie=PluginEstimate()
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)
	labels=Labels(double(indata['classifier_labels']))
	pie.train(feats['train'], labels)

	fun=eval(indata['name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], pie)
	km_train=max(abs(indata['km_train']-kernel.get_kernel_matrix()).flat)

	kernel.init(feats['train'], feats['test'])
	pie.set_testfeatures(feats['test'])
	pie.test()
	km_test=max(abs(indata['km_test']-kernel.get_kernel_matrix()).flat)
	classified=max(abs(
		pie.classify().get_labels()-indata['classifier_classified']))

	return util.check_accuracy(indata['accuracy'],
		km_train=km_train, km_test=km_test, classified=classified)

def _kernel_top_fisher (indata):
	feats={}
	wordfeats=util.get_feats_string_complex(indata)

	pos=HMM(wordfeats['train'], indata['N'], indata['M'],
		indata['pseudo'])
	pos.train()
	pos.baum_welch_viterbi_train(BW_NORMAL)
	neg=HMM(wordfeats['train'], indata['N'], indata['M'],
		indata['pseudo'])
	neg.train()
	neg.baum_welch_viterbi_train(BW_NORMAL)
	pos_clone=HMM(pos)
	neg_clone=HMM(neg)
	pos_clone.set_observations(wordfeats['test'])
	neg_clone.set_observations(wordfeats['test'])

	fun=eval(indata['name_features']+'Features')
	if indata['name_features']=='TOP':
		feats['train']=fun(10, pos, neg, False, False)
		feats['test']=fun(10, pos_clone, neg_clone, False, False)
	else:
		feats['train']=fun(10, pos, neg)
		feats['train'].set_opt_a(-1) #estimate prior
		feats['test']=fun(10, pos_clone, neg_clone)
		feats['test'].set_a(feats['train'].get_a()) #use prior from training data

	args=util.get_args(indata, 'kernel_arg')
	kernel=LinearKernel(feats['train'], feats['train'], *args)
	km_train=max(abs(
		indata['km_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata['km_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata['accuracy'],
		km_train=km_train, km_test=km_test)

########################################################################
# public
########################################################################

def test (indata):
	# FIXME: name_features has to go away, a bit tricky, though
	if indata.has_key('name_features'):
		return _kernel_top_fisher(indata)

	names=['Combined', 'AUC', 'Custom']
	for name in names:
		if indata['name']==name:
			return eval('_kernel_'+name.lower()+'(indata)')

	names=['HistogramWord', 'SalzbergWord']
	for name in names:
		if indata['name']==name:
			return _kernel_pie(indata)

	return _kernel(indata)

