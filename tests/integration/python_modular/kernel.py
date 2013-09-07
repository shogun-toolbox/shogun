"""
Test Kernel
"""

from shogun.Features import *
from shogun.Kernel import *
from shogun.Preprocessor import *
from shogun.Distance import *
from shogun.Classifier import PluginEstimate
from shogun.Distribution import HMM, BW_NORMAL
from numpy import array, ushort, ubyte, double

import util

########################################################################
# kernel computation
########################################################################

def _evaluate (indata, prefix):
	feats=util.get_features(indata, prefix)
	kfun=eval(indata[prefix+'name']+'Kernel')
	kargs=util.get_args(indata, prefix)
	kernel=kfun(*kargs)
	if prefix+'normalizer' in indata:
		kernel.set_normalizer(eval(indata[prefix+'normalizer']+'()'))

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], km_train=km_train, km_test=km_test)


def _get_subkernels (indata, prefix):
	subkernels={}
	prefix=prefix+'subkernel'
	len_prefix=len(prefix)

	# loop through indata (unordered dict) to gather subkernel data
	for key in indata:
		if key.find(prefix)==-1:
			continue

		# get subkernel's number
		try:
			num=key[len_prefix]
		except ValueError:
			raise ValueError('Cannot find number for subkernel: "%s"!' % data)

		# get item's name
		name=key[len_prefix+2:]

		# append new item
		if num not in subkernels:
			subkernels[num]={}
		subkernels[num][name]=indata[key]

	# got all necessary information in new structure, now create a kernel
	# object for each subkernel
	for num, data in subkernels.items():
		fun=eval(data['name']+'Kernel')
		args=util.get_args(data, '')
		subkernels[num]['kernel']=fun(*args)

	return subkernels


def _evaluate_combined (indata, prefix):
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}

	subkernels=_get_subkernels(indata, prefix)
	for subk in subkernels.values():
		feats_subk=util.get_features(subk, '')
		feats['train'].append_feature_obj(feats_subk['train'])
		feats['test'].append_feature_obj(feats_subk['test'])
		kernel.append_kernel(subk['kernel'])

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		indata['kernel_matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata['kernel_matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata[prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


def _evaluate_auc (indata, prefix):
	subk=_get_subkernels(indata, prefix)['0']
	feats_subk=util.get_features(subk, '')
	subk['kernel'].init(feats_subk['train'], feats_subk['test'])

	feats={
		'train': WordFeatures(indata[prefix+'data_train'].astype(ushort)),
		'test': WordFeatures(indata[prefix+'data_test'].astype(ushort))
	}
	kernel=AUCKernel(10, subk['kernel'])

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata[prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


def _evaluate_custom (indata, prefix):
	feats={
		'train': RealFeatures(indata[prefix+'data']),
		'test': RealFeatures(indata[prefix+'data'])
	}

	symdata=indata[prefix+'symdata']
	lowertriangle=array([symdata[(x,y)] for x in range(symdata.shape[1])
		for y in range(symdata.shape[0]) if y<=x])
	kernel=CustomKernel()
	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	triangletriangle=max(abs(
		indata[prefix+'matrix_triangletriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_triangle_kernel_matrix_from_full(indata[prefix+'symdata'])
	fulltriangle=max(abs(
		indata[prefix+'matrix_fulltriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_full_kernel_matrix_from_full(indata[prefix+'data'])
	fullfull=max(abs(
		indata[prefix+'matrix_fullfull']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata[prefix+'accuracy'],
		triangletriangle=triangletriangle, fulltriangle=fulltriangle,
		fullfull=fullfull)


def _evaluate_pie (indata, prefix):
	pie=PluginEstimate()
	feats=util.get_features(indata, prefix)
	labels=BinaryLabels(double(indata['classifier_labels']))
	pie.set_labels(labels)
	pie.set_features(feats['train'])
	pie.train()

	fun=eval(indata[prefix+'name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], pie)
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)

	kernel.init(feats['train'], feats['test'])
	pie.set_features(feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)
	classified=max(abs(
		pie.apply().get_values()-indata['classifier_classified']))

	return util.check_accuracy(indata[prefix+'accuracy'],
		km_train=km_train, km_test=km_test, classified=classified)


def _evaluate_top_fisher (indata, prefix):
	feats={}
	wordfeats=util.get_features(indata, prefix)

	pos_train=HMM(wordfeats['train'], indata[prefix+'N'], indata[prefix+'M'],
		indata[prefix+'pseudo'])
	pos_train.train()
	pos_train.baum_welch_viterbi_train(BW_NORMAL)
	neg_train=HMM(wordfeats['train'], indata[prefix+'N'], indata[prefix+'M'],
		indata[prefix+'pseudo'])
	neg_train.train()
	neg_train.baum_welch_viterbi_train(BW_NORMAL)
	pos_test=HMM(pos_train)
	pos_test.set_observations(wordfeats['test'])
	neg_test=HMM(neg_train)
	neg_test.set_observations(wordfeats['test'])

	if indata[prefix+'name']=='TOP':
		feats['train']=TOPFeatures(10, pos_train, neg_train, False, False)
		feats['test']=TOPFeatures(10, pos_test, neg_test, False, False)
	else:
		feats['train']=FKFeatures(10, pos_train, neg_train)
		feats['train'].set_opt_a(-1) #estimate prior
		feats['test']=FKFeatures(10, pos_test, neg_test)
		feats['test'].set_a(feats['train'].get_a()) #use prior from training data

	prefix='kernel_'
	args=util.get_args(indata, prefix)
	kernel=PolyKernel(feats['train'], feats['train'], *args)
#	kernel=PolyKernel(*args)
#	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata[prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


########################################################################
# public
########################################################################

def test (indata):
	prefix='topfk_'
	if prefix+'name' in indata:
		return _evaluate_top_fisher(indata, prefix)

	prefix='kernel_'
	names=['Combined', 'AUC', 'Custom']
	for name in names:
		if indata[prefix+'name']==name:
			return eval('_evaluate_'+name.lower()+'(indata, prefix)')

	names=['HistogramWordString', 'SalzbergWordString']
	for name in names:
		if indata[prefix+'name']==name:
			return _evaluate_pie(indata, prefix)

	return _evaluate(indata, prefix)

