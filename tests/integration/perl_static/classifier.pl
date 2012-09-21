"""
Test Classifier
"""

from numpy import double
from sg import sg

import util


def _set_classifier (indata, prefix):
	if indata.has_key(prefix+'labels'):
		sg('set_labels', 'TRAIN', double(indata[prefix+'labels']))

	cname=util.fix_classifier_name_inconsistency(indata[prefix+'name'])
	sg('new_classifier', cname)

	if indata.has_key(prefix+'bias'):
		sg('svm_use_bias', True)
	else:
		sg('svm_use_bias', False)

	if indata.has_key(prefix+'epsilon'):
		sg('svm_epsilon', indata[prefix+'epsilon'])
	if indata.has_key(prefix+'max_train_time'):
		sg('svm_max_train_time', indata[prefix+'max_train_time'])
	if indata.has_key(prefix+'linadd_enabled'):
		sg('use_linadd', True)
	if indata.has_key(prefix+'batch_enabled'):
		sg('use_batch_computation', True)
	if indata.has_key(prefix+'num_threads'):
		sg('threads', indata[prefix+'num_threads'])


def _train (indata, prefix):
	if indata[prefix+'type']=='knn':
		sg('train_classifier', indata[prefix+'k'])
	elif indata[prefix+'type']=='lda':
		sg('train_classifier', indata[prefix+'gamma'])
	elif indata[prefix+'type']=='perceptron':
		# does not converge
		try:
			sg('train_classifier')
		except RuntimeError:
			import sys
			sys.exit(0)
	else:
		if indata.has_key(prefix+'C'):
			sg('c', double(indata[prefix+'C']))
		sg('train_classifier')


def _get_alpha_and_sv(indata, prefix):
	if not indata.has_key(prefix+'alpha_sum') and \
		not indata.has_key(prefix+'sv_sum'):
		return None, None

	a=0
	sv=0
	if indata.has_key(prefix+'label_type') and \
		indata[prefix+'label_type']=='series':
		for i in xrange(sg('get_num_svms')):
			[dump, weights]=sg('get_svm', i)
			weights=weights.T
			for item in weights[0].tolist():
				a+=item
			for item in weights[1].tolist():
				sv+=item
		a=abs(a-indata[prefix+'alpha_sum'])
		sv=abs(sv-indata[prefix+'sv_sum'])
	else:
		[dump, weights]=sg('get_svm')
		weights=weights.T
		for item in weights[0].tolist():
			a+=item
		a=abs(a-indata[prefix+'alpha_sum'])
		for item in weights[1].tolist():
			sv+=item
		sv=abs(sv-indata[prefix+'sv_sum'])

	return a, sv


def _evaluate (indata, prefix):
	alphas=0
	bias=0
	sv=0

	if indata[prefix+'type']=='lda':
		pass
	else:
		if indata.has_key(prefix+'label_type') and \
			indata[prefix+'label_type'] != 'series' and \
			indata.has_key(prefix+'bias'):
			[b, weights]=sg('get_svm')
			weights=weights.T
			bias=abs(b-indata[prefix+'bias'])

		alphas, sv=_get_alpha_and_sv(indata, prefix)

	classified=max(abs(sg('classify')-indata[prefix+'classified']))

	return util.check_accuracy(indata[prefix+'accuracy'],
		alphas=alphas, bias=bias, support_vectors=sv, classified=classified)

########################################################################
# public
########################################################################

def test (indata):
	prefix='classifier_'

	if indata[prefix+'type']=='kernel':
		feature_prefix='kernel_'
	elif indata[prefix+'type']=='knn':
		feature_prefix='distance_'
	else:
		feature_prefix='classifier_'

	try:
		util.set_features(indata, feature_prefix)
	except NotImplementedError, e:
		print e
		return True

	if indata[prefix+'type']=='kernel':
		util.set_and_train_kernel(indata)
	elif indata[prefix+'type']=='knn':
		util.set_and_train_distance(indata)

	try:
		_set_classifier(indata, prefix)
	except RuntimeError:
		print "%s is disabled/unavailable!" % indata[prefix+'name']
		return True

	_train(indata, prefix)

	return _evaluate(indata, prefix)

