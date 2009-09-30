"""
Test Kernel
"""

from sg import sg
from numpy import double
import util


def _evaluate_kernel (indata, prefix, kernel_is_set=False):
	if not kernel_is_set:
		util.set_and_train_kernel(indata)

	kmatrix=sg('get_kernel_matrix', 'TRAIN')
	km_train=max(abs(indata[prefix+'matrix_train']-kmatrix).flat)

	kmatrix=sg('get_kernel_matrix', 'TEST')
	km_test=max(abs(indata[prefix+'matrix_test']-kmatrix).flat)

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
			raise ValueError, 'Cannot find number for subkernel: "%s"!' % data

		# get item's name
		name=key[len_prefix+2:]

		# append new item
		if not subkernels.has_key(num):
			subkernels[num]={}
		subkernels[num][name]=indata[key]

	# got all necessary information in new structure, now create a kernel
	# object for each subkernel
	for num, data in subkernels.iteritems():
		subkernels[num]['args']=util.get_args(data, '')
		subkernels[num]['name']=util.fix_kernel_name_inconsistency(data['name'])

	return subkernels


def _evaluate_combined (indata, prefix):
	sg('set_kernel', 'COMBINED', util.SIZE_CACHE)

	subkernels=_get_subkernels(indata, prefix)
	for subk in subkernels.itervalues():
		sg('add_kernel', 1., subk['name'], subk['feature_type'].upper(),
			*subk['args'])

		if subk.has_key('alphabet'):
			sg('add_features', 'TRAIN',
				list(subk['data_train'][0]), subk['alphabet'])
			sg('add_features', 'TEST',
				list(subk['data_test'][0]), subk['alphabet'])
		else:
			sg('add_features', 'TRAIN', subk['data_train'][0])
			sg('add_features', 'TEST', subk['data_test'][0])

	return _evaluate_kernel(indata, prefix, True)


def _evaluate_auc (indata, prefix):
	raise NotImplementedError, 'AUC kernel not yet supported in static interfaces.'

	subk=_get_subkernels(indata, prefix)[0]
	subk['kernel'].init(feats_subk['train'], feats_subk['test'])

	feats={
		'train': WordFeatures(indata['data_train'].astype(ushort)),
		'test': WordFeatures(indata['data_test'].astype(ushort))}
	kernel=AUCKernel(10, subk['kernel'])

	return _evaluate_kernel(indata, prefix, True)


def _evaluate_custom (indata, prefix):
	raise NotImplementedError, 'Custom kernel not yet implemented in static interfaces.'

	symdata=indata[prefix+'symdata']
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])

	sg('set_kernel', 'CUSTOM')
	sg('set_triangle_kernel_matrix_from_triangle', lowertriangle)
	triangletriangle=max(abs(
		indata[prefix+'matrix_triangletriangle']-sg('get_kernel_matrix')).flat)

	sg('set_triangle_kernel_matrix_from_full', indata[prefix+'symdata'])
	fulltriangle=max(abs(
		indata[prefix+'matrix_fulltriangle']-sg('get_kernel_matrix')).flat)

	sg('set_full_kernel_matrix_from_full', indata[prefix+'data'])
	fullfull=max(abs(
		indata[prefix+'matrix_fullfull']-sg('get_kernel_matrix')).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'],
		triangletriangle=triangletriangle,
		fulltriangle=fulltriangle,
		fullfull=fullfull
	)


def _evaluate_pie (indata, prefix):
	pseudo_pos=1e-10
	pseudo_neg=1e-10

	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)

	sg('set_labels', 'TRAIN', double(indata['classifier_labels']))
	sg('train_estimator')

	return _evaluate_kernel(indata, prefix)


def _evaluate_top_fisher (indata, prefix):
	raise NotImplementedError, 'TOP/Fisher not yet supported in static interfaces.'

	sg('new_hmm', indata[prefix+'N'], indata[prefix+'M'])
	pos=HMM(wordfeats['train'], indata[prefix+'N'], indata[prefix+'M'],
		indata[prefix+'pseudo'])
	pos.train()
	pos.baum_welch_viterbi_train(BW_NORMAL)
	neg=HMM(wordfeats['train'], indata[prefix+'N'], indata[prefix+'M'],
		indata[prefix+'pseudo'])
	neg.train()
	neg.baum_welch_viterbi_train(BW_NORMAL)
	pos_clone=HMM(pos)
	neg_clone=HMM(neg)
	pos_clone.set_observations(wordfeats['test'])
	neg_clone.set_observations(wordfeats['test'])

	if indata[prefix+'type']=='TOP':
		feats['train']=TOPFeatures(10, pos, neg, False, False)
		feats['test']=TOPFeatures(10, pos_clone, neg_clone, False, False)
	else:
		feats['train']=FKFeatures(10, pos, neg)
		feats['train'].set_opt_a(-1) #estimate prior
		feats['test']=FKFeatures(10, pos_clone, neg_clone)
		feats['test'].set_a(feats['train'].get_a()) #use prior from training data

	prefix='kernel_'
	args=util.get_args(indata, prefix)
	kernel=PolyKernel(feats['train'], feats['train'], *args)
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
	if indata.has_key(prefix+'name'):
		try:
			util.set_features(indata, prefix)
			return _evaluate_top_fisher(indata, prefix)
		except NotImplementedError, e:
			print e
			return True

	prefix='kernel_'
	try:
		util.set_features(indata, prefix)
		names=['Combined', 'AUC', 'Custom']
		for name in names:
			if indata[prefix+'name']==name:
				return eval('_evaluate_'+name.lower()+'(indata, prefix)')

		names=['HistogramWordString', 'SalzbergWordString']
		for name in names:
			if indata[prefix+'name']==name:
				return _evaluate_pie(indata, prefix)

		# pretty normal kernel
		return _evaluate_kernel(indata, prefix)
	except NotImplementedError, e:
		print e
		return True


