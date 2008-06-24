"""
Test Classifier
"""

from numpy import double
from sg import sg

import util



def _set_features (indata):
	# set features
	if indata.has_key('alphabet'):
		sg('set_features', 'TRAIN',
			list(indata['data_train'][0]), indata['alphabet'])
		sg('set_features', 'TEST',
			list(indata['data_test'][0]), indata['alphabet'])
	else:
		sg('set_features', 'TRAIN',
			indata['data_train'].astype(eval(indata['data_type'])))
		sg('set_features', 'TEST',
			indata['data_test'].astype(eval(indata['data_type'])))


def _set_machine (indata):
	cache_size=10

	if indata['classifier_type']=='kernel':
		kargs=util.get_args(indata, 'kernel_arg')
		kname=util.fix_kname_inconsistency(indata['kernel_name'])

		if kname=='COMMSTRING':
			kargs[1]=util.fix_normalization_inconsistency(kargs[1])
			util.convert_features_and_add_preproc(indata)

		sg('set_kernel', kname, indata['feature_type'].upper(),
			cache_size, *kargs)
		sg('init_kernel', 'TRAIN')

	elif indata['classifier_type']=='knn':
		dargs=util.get_args(indata, 'distance_arg')
		dname=util.fix_dname_inconsistency(indata['distance_name'])
		sg('set_distance', dname, indata['feature_type'].upper(),
			cache_size, *dargs)
		sg('init_distance', 'TRAIN')


def _set_classifier (indata):
	cname=util.fix_cname_inconsistency(indata['name'])
	sg('new_classifier', cname)
	if indata.has_key('classifier_bias'):
		sg('svm_use_bias', True)
	else:
		sg('svm_use_bias', False)
	if indata.has_key('classifier_epsilon'):
		sg('svm_epsilon', indata['classifier_epsilon'])
	if indata.has_key('classifier_tube_epsilon'):
		sg('svr_tube_epsilon', indata['classifier_tube_epsilon'])
	if indata.has_key('classifier_max_train_time'):
		sg('svm_max_train_time', indata['classifier_max_train_time'])
	if indata.has_key('classifier_linadd_enabled'):
		sg('use_linadd', True)
	if indata.has_key('classifier_batch_enabled'):
		sg('use_batch_computation', True)


def _train (indata):
	if indata['classifier_type']=='knn':
		sg('train_classifier', indata['classifier_k'])
	elif indata['classifier_type']=='lda':
		sg('train_classifier', indata['classifier_gamma'])
	else:
		if indata.has_key('classifier_C'):
			sg('c', double(indata['classifier_C']))
		sg('train_classifier')


def _evaluate (indata):
	res={
		'alphas':0,
		'bias':0,
		'sv':0,
		'accuracy':indata['classifier_accuracy'],
	}

	if indata['classifier_type']=='knn':
		sg('init_distance', 'TEST')
	elif indata['classifier_type']=='lda':
		pass
	else:
		[bias, weights]=sg('get_svm')
		weights=weights.T
		if indata.has_key('classifier_bias'):
			res['bias']=abs(bias[0][0]-indata['classifier_bias'])
		if indata.has_key('classifier_alphas'):
			res['alphas']=max(abs(weights[0]-indata['classifier_alphas']))
		if indata.has_key('classifier_support_vectors'):
			res['sv']=max(abs(weights[1]-indata['classifier_support_vectors']))
		sg('init_kernel', 'TEST')

	res['classified']=max(abs(sg('classify')-indata['classifier_classified']))

	return util.check_accuracy(res['accuracy'],
		alphas=res['alphas'], bias=res['bias'], sv=res['sv'],
		classified=res['classified'])

########################################################################
# public
########################################################################

def test (indata):
	if indata.has_key('classifier_type') and indata['classifier_type']=='linear':
		print "Sparse features / Linear classifiers not supported yet!"
		return False

	_set_features(indata)
	_set_machine(indata)

	# set labels
	if indata.has_key('classifier_labels'):
		sg('set_labels', 'TRAIN', double(indata['classifier_labels']))

	# set misc
	if indata.has_key('classifier_num_threads'):
		sg('threads', indata['classifier_num_threads'])

	try:
		_set_classifier(indata)
	except RuntimeError:
		print "%s is disabled/unavailable!" % indata['name']
		return False

	_train(indata)
	return _evaluate(indata)

