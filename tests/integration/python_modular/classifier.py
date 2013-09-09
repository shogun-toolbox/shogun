"""
Test Classifier
"""

from numpy import double
from modshogun import BinaryLabels
from modshogun import *
from modshogun import *
from modshogun import *

import util

def _get_machine (indata, prefix, feats):
	if indata[prefix+'type']=='kernel':
		pre='kernel_'
		kargs=util.get_args(indata, pre)
		kfun=eval(indata[pre+'name']+'Kernel')
		machine=kfun(feats['train'], feats['train'], *kargs)

		if indata[pre+'name']=='Linear':
			normalizer=eval(indata[pre+'normalizer']+'()')
			machine.set_normalizer(normalizer)
			machine.init(feats['train'], feats['train'])

		machine.parallel.set_num_threads(indata[prefix+'num_threads'])
	elif indata[prefix+'type']=='knn':
		pre='distance_'
		dargs=util.get_args(indata, pre)
		dfun=eval(indata[pre+'name'])
		machine=dfun(feats['train'], feats['train'], *dargs)
		machine.parallel.set_num_threads(indata[prefix+'num_threads'])
	else:
		machine=None

	return machine


def _get_results_alpha_and_sv(indata, prefix, classifier):
	if prefix+'alpha_sum' not in indata and \
		prefix+'sv_sum' not in indata:
		return None, None

	a=0
	sv=0
	if prefix+'label_type' in indata and \
		indata[prefix+'label_type']=='series':
		for i in range(classifier.get_num_svms()):
			subsvm=classifier.get_svm(i)
			for item in subsvm.get_alphas().tolist():
				a+=item
			for item in subsvm.get_support_vectors().tolist():
				sv+=item

		a=abs(a-indata[prefix+'alpha_sum'])
		sv=abs(sv-indata[prefix+'sv_sum'])
	else:
		for item in classifier.get_alphas().tolist():
			a+=item
		a=abs(a-indata[prefix+'alpha_sum'])
		for item in classifier.get_support_vectors().tolist():
			sv+=item
		sv=abs(sv-indata[prefix+'sv_sum'])

	return a, sv


def _get_results (indata, prefix, classifier, machine=None, feats=None):
	res={
		'alphas':0,
		'bias':0,
		'sv':0,
		'accuracy':indata[prefix+'accuracy'],
	}

	if prefix+'bias' in indata:
		res['bias']=abs(classifier.get_bias()-indata[prefix+'bias'])

	res['alphas'], res['sv']=_get_results_alpha_and_sv(
		indata, prefix, classifier)

	ctype=indata[prefix+'type']
	if ctype=='kernel' or ctype=='knn':
		machine.init(feats['train'], feats['test'])
	else:
		ctypes=('linear', 'perceptron', 'lda', 'wdsvmocas')
		if ctype in ctypes:
			classifier.set_features(feats['test'])

	res['classified']=max(abs(
		classifier.apply().get_values()-indata[prefix+'classified']))
	return res


def _evaluate (indata):
	prefix='classifier_'
	ctype=indata[prefix+'type']
	if indata[prefix+'name']=='KNN':
		feats=util.get_features(indata, 'distance_')
	elif ctype=='kernel':
		feats=util.get_features(indata, 'kernel_')
	else:
		feats=util.get_features(indata, prefix)

	machine=_get_machine(indata, prefix, feats)

	try:
		fun=eval(indata[prefix+'name'])
	except NameError as e:
		print("%s is disabled/unavailable!"%indata[prefix+'name'])
		return False

	# cannot refactor into function, because labels is unrefed otherwise
	if prefix+'labels' in indata:
		labels=BinaryLabels(double(indata[prefix+'labels']))
		if ctype=='kernel':
			classifier=fun(indata[prefix+'C'], machine, labels)
		elif ctype=='linear':
			classifier=fun(indata[prefix+'C'], feats['train'], labels)
		elif ctype=='knn':
			classifier=fun(indata[prefix+'k'], machine, labels)
		elif ctype=='lda':
			classifier=fun(indata[prefix+'gamma'], feats['train'], labels)
		elif ctype=='perceptron':
			classifier=fun(feats['train'], labels)
		elif ctype=='wdsvmocas':
			classifier=fun(indata[prefix+'C'], indata[prefix+'degree'],
				indata[prefix+'degree'], feats['train'], labels)
		else:
			return False
	else:
		classifier=fun(indata[prefix+'C'], machine)

	if classifier.get_name() == 'LibLinear':
		print(classifier.get_name(), "yes")
		classifier.set_liblinear_solver_type(L2R_LR)

	classifier.parallel.set_num_threads(indata[prefix+'num_threads'])
	if ctype=='linear':
		if prefix+'bias' in indata:
			classifier.set_bias_enabled(True)
		else:
			classifier.set_bias_enabled(False)
	if ctype=='perceptron':
		classifier.set_learn_rate=indata[prefix+'learn_rate']
		classifier.set_max_iter=indata[prefix+'max_iter']
	if prefix+'epsilon' in indata:
		try:
			classifier.set_epsilon(indata[prefix+'epsilon'])
		except AttributeError:
			pass
	if prefix+'max_train_time' in indata:
		classifier.set_max_train_time(indata[prefix+'max_train_time'])
	if prefix+'linadd_enabled' in indata:
		classifier.set_linadd_enabled(indata[prefix+'linadd_enabled'])
	if prefix+'batch_enabled' in indata:
		classifier.set_batch_computation_enabled(indata[prefix+'batch_enabled'])

	classifier.train()

	res=_get_results(indata, prefix, classifier, machine, feats)
	return util.check_accuracy(res['accuracy'],
		alphas=res['alphas'], bias=res['bias'], sv=res['sv'],
		classified=res['classified'])


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

