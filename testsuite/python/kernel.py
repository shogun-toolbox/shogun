"""
Test Kernel
"""

from sg import sg
from numpy import double
import util


def _evaluate (indata):
	try:
		# FIXME: name_features has to go away, a bit tricky, though
		if indata.has_key('name_features'):
			return _evaluate_top_fisher(indata)

		names=['Combined', 'AUC', 'Custom']
		for name in names:
			if indata['name']==name:
				return eval('_evaluate_'+name.lower()+'(indata)')

		names=['HistogramWord', 'SalzbergWord']
		for name in names:
			if indata['name']==name:
				return _evaluate_pie(indata)
	except NotImplementedError, e:
		print e
		return True

	# pretty normal kernel
	return _evaluate_kernel(indata)


def _evaluate_kernel (indata):
	util.set_and_train_kernel(indata)

	kmatrix=sg('get_kernel_matrix')
	ktrain=max(abs(indata['km_train']-kmatrix).flat)

	sg('init_kernel', 'TEST')
	kmatrix=sg('get_kernel_matrix')
	ktest=max(abs(indata['km_test']-kmatrix).flat)

	return util.check_accuracy(indata['accuracy'], ktrain=ktrain, ktest=ktest)


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

	for idx, subk in enumerate(subkernels):
		subkernels[idx]['args']=util.get_args(subk, 'kernel_arg')
		subkernels[idx]['name']=util.fix_kernel_name_inconsistency(subk['name'])

	return subkernels


def _evaluate_combined (indata):
	sg('set_kernel', 'COMBINED', util.SIZE_CACHE)

	subkernels=_get_subkernels(indata)
	for subk in subkernels:
		sg('add_kernel', 1., subk['name'], subk['feature_type'].upper(), *subk['args'])

		if subk.has_key('alphabet'):
			sg('add_features', 'TRAIN', list(subk['data_train'][0]), subk['alphabet'])
			sg('add_features', 'TEST', list(subk['data_test'][0]), subk['alphabet'])
		else:
			sg('add_features', 'TRAIN', subk['data_train'][0])
			sg('add_features', 'TEST', subk['data_test'][0])

	return _evaluate_subkernels(indata)


def _evaluate_auc (indata):
	raise NotImplementedError, 'AUC kernel not yet supported in static interfaces.'

	subk=_get_subkernels(indata)[0]
	subk['kernel'].init(feats_subk['train'], feats_subk['test'])

	feats={
		'train':WordFeatures(indata['data_train'].astype(eval(indata['data_type']))),
		'test':WordFeatures(indata['data_test'].astype(eval(indata['data_type'])))}
	kernel=AUCKernel(10, subk['kernel'])

	return _evaluate_subkernels(indata)


def _evaluate_subkernels (indata):
	sg('init_kernel', 'TRAIN')
	km_train=max(abs(indata['km_train']-sg('get_kernel_matrix')).flat)
	sg('init_kernel', 'TEST')
	km_test=max(abs(indata['km_test']-sg('get_kernel_matrix')).flat)

	return util.check_accuracy(indata['accuracy'],
		km_train=km_train, km_test=km_test)


def _evaluate_custom (indata):
	raise NotImplementedError, 'Custom kernel not yet implemented in static interfaces.'

	symdata=indata['symdata']
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])

	sg('set_kernel', 'CUSTOM')
	sg('set_triangle_kernel_matrix_from_triangle', lowertriangle)
	triangletriangle=max(abs(
		indata['km_triangletriangle']-sg('get_kernel_matrix')).flat)

	sg('set_triangle_kernel_matrix_from_full', indata['symdata'])
	fulltriangle=max(abs(
		indata['km_fulltriangle']-sg('get_kernel_matrix')).flat)

	sg('set_full_kernel_matrix_from_full', indata['data'])
	fullfull=max(abs(indata['km_fullfull']-sg('get_kernel_matrix')).flat)

	return util.check_accuracy(indata['accuracy'],
		triangletriangle=triangletriangle, fulltriangle=fulltriangle,
		fullfull=fullfull)

def _evaluate_pie (indata):
	pseudo_pos=1e-10
	pseudo_neg=1e-10

	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)

	sg('set_labels', 'TRAIN', double(indata['classifier_labels']))
	sg('train_estimator')
	util.set_and_train_kernel(indata)

	km_train=max(abs(indata['km_train']-sg('get_kernel_matrix')).flat)

	sg('init_kernel', 'TEST')
	km_test=max(abs(indata['km_test']-sg('get_kernel_matrix')).flat)

	return util.check_accuracy(indata['accuracy'],
		km_train=km_train, km_test=km_test)

def _evaluate_top_fisher (indata):
	raise NotImplementedError, 'TOP/Fisher not yet supported in static interfaces.'

	sg('new_hmm', indata['distribution_N'], indata['distribution_M'])
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

	if indata['name_features']=='TOP':
		feats['train']=TOPFeatures(10, pos, neg, False, False)
		feats['test']=TOPFeatures(10, pos_clone, neg_clone, False, False)
	else:
		feats['train']=FKFeatures(10, pos, neg)
		feats['train'].set_opt_a(-1) #estimate prior
		feats['test']=FKFeatures(10, pos_clone, neg_clone)
		feats['test'].set_a(feats['train'].get_a()) #use prior from training data

	args=util.get_args(indata, 'kernel_arg')
	kernel=PolyKernel(feats['train'], feats['train'], *args)
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
	try:
		util.set_features(indata)
	except NotImplementedError, e:
		print e
		return True

	return _evaluate(indata)
