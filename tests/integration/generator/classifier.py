"""Generator for Classifier"""

import numpy
import shogun.Library as library
import shogun.Classifier as classifier
from shogun.Kernel import *
from shogun.Distance import EuclidianDistance
from shogun.Features import Labels, RAWDNA

import fileop
import featop
import dataop
import category


##########################################################################
# svm
##########################################################################

def _get_svm (params, labels, feats, kernel):
	"""Return an SVM object.

	This function instantiates an SVM depending on the parameters.

	@param name Name of the SVM to instantiate
	@param labels Labels to be used for the SVM (if at all!)
	@param params Misc parameters for the SVM's constructor
	@return An SVM object
	"""

	try:
		svm=eval('classifier.'+params['name'])
	except AttributeError, e:
		return False

	if params['type']=='kernel':
		kernel.parallel.set_num_threads(params['num_threads'])
		kernel.init(feats['train'], feats['train'])

		if labels is None:
			return svm(params['C'], kernel)
		else:
			return svm(params['C'], kernel, labels)
	elif params['type']=='wdsvmocas':
		return svm(params['C'], params['degree'], params['degree'],
			feats['train'], labels)
	else:
		return svm(params['C'], feats['train'], labels)


def _get_svm_sum_alpha_and_sv (svm, ltype):
	"""Return sums of alphas and support vectors for given (MultiClass) SVM.
		Since alphas and support vectors are only used for comparison, it is
		alright to only regard their sums. Especially in case of
		MultiClassSVMs it makes test files smaller and easier to comprehend.

	@param svm (MultiClass) SVM to retrieve alphas from
	@param ltype label type of SVM
	@return 2-element list with sums of alphas and support vectors
	"""

	a=0
	sv=0
	if ltype=='series':
		for i in xrange(svm.get_num_svms()):
			subsvm=svm.get_svm(i)
			for item in subsvm.get_alphas().tolist():
				a+=item
			for item in subsvm.get_support_vectors().tolist():
				sv+=item
	else:
		for item in svm.get_alphas().tolist():
			a+=item
		for item in svm.get_support_vectors().tolist():
			sv+=item

	return a, sv


def _compute_svm (params, labels, feats, kernel, pout):
	"""Perform computations on SVM.

	Perform all necessary computations on SVM and gather the output.

	@param params misc parameters for the SVM's constructor
	@param labels labels to be used for the SVM (if at all)
	@param feats features to the SVM
	@param kernel kernel for kernel-SVMs
	@param pout previously gathered output data ready to be written to file
	"""

	svm=_get_svm(params, labels, feats, kernel)
	if not svm:
		return

	svm.parallel.set_num_threads(params['num_threads'])
	try:
		svm.set_epsilon(params['epsilon'])
	except AttributeError: #SGD does not have an accuracy parameter
		pass

	if params.has_key('bias_enabled'):
		svm.set_bias_enabled(params['bias_enabled'])
	if params.has_key('max_train_time'):
		svm.set_max_train_time(params['max_train_time'])
		params['max_train_time']=params['max_train_time']
	if params.has_key('linadd_enabled'):
		svm.set_linadd_enabled(params['linadd_enabled'])
	if params.has_key('batch_enabled'):
		svm.set_batch_computation_enabled(params['batch_enabled'])

	svm.train()

	if ((params.has_key('bias_enabled') and params['bias_enabled']) or
		params['type']=='kernel'):
		params['bias']=svm.get_bias()

	if params['type']=='kernel':
		params['alpha_sum'], params['sv_sum']= \
			_get_svm_sum_alpha_and_sv(svm, params['label_type'])
		kernel.init(feats['train'], feats['test'])
	elif params['type']=='linear' or params['type']=='wdsvmocas':
		svm.set_features(feats['test'])

	params['classified']=svm.apply().get_labels()

	output=fileop.get_output(category.CLASSIFIER, params)
	if pout:
		output.update(pout)
	fileop.write(category.CLASSIFIER, output)


def _loop_svm (svms, params, feats, kernel=None, pout=None):
	"""Loop through SVM computations, only slightly differing in parameters.

	Loop through SVM computations with little variations in the parameters for
	the SVM. Not necessarily used by all SVMs in this generator.

	@param svms tuple of SVM names to loop through
	@param params parameters to the SVMs
	@param feats features to the SVMs
	@param kernel kernel for kernel-based SVMs
	@param pout previously gathered output data ready to be written to file
	"""

	for name in svms:
		parms={ 'name': name, 'num_threads': 1, 'C': .017, 'epsilon': 1e-5 }
		parms['accuracy']=parms['epsilon']*10
		parms.update(params)

		if params['label_type'] is not None:
			parms['labels'], labels=dataop.get_labels(
				feats['train'].get_num_vectors(), params['label_type'])
		else:
			labels=None

		_compute_svm(parms, labels, feats, kernel, pout)
		parms['C']=.23
		_compute_svm(parms, labels, feats, kernel, pout)
		parms['C']=1.5
		_compute_svm(parms, labels, feats, kernel, pout)
		parms['C']=30
		_compute_svm(parms, labels, feats, kernel, pout)

		if params['type']=='kernel':
			_compute_svm(parms, labels, feats, kernel, pout)

		parms['num_threads']=16
		_compute_svm(parms, labels, feats, kernel, pout)


def _run_svm_kernel ():
	"""Run all kernel-based SVMs."""

	kparams={
		'name': 'Gaussian',
		'args': {'key': ('width',), 'val': (1.5,)},
		'feature_class': 'simple',
		'feature_type': 'Real',
		'data': dataop.get_clouds(2)
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=GaussianKernel(10, *kparams['args']['val'])
	output=fileop.get_output(category.KERNEL, kparams)

	svms=('SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM')
	params={
		'type': 'kernel',
		'label_type': 'twoclass'
	}
	_loop_svm(svms, params, feats, kernel, output)

	svms=('LibSVMOneClass',)
	params['label_type']=None
	_loop_svm(svms, params, feats, kernel, output)

	svms=('LibSVMMultiClass', 'GMNPSVM')
	params['label_type']='series'
	kparams['data']=dataop.get_clouds(3)
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	output=fileop.get_output(category.KERNEL, kparams)
	_loop_svm(svms, params, feats, kernel, output)

	svms=('SVMLight', 'GPBTSVM')
	params['label_type']='twoclass'
	kparams={
		'name': 'Linear',
		'feature_class': 'simple',
		'feature_type': 'Real',
		'data': dataop.get_clouds(2),
		'normalizer': AvgDiagKernelNormalizer()
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=LinearKernel()
	kernel.set_normalizer(kparams['normalizer'])
	output=fileop.get_output(category.KERNEL, kparams)
	_loop_svm(svms, params, feats, kernel, output)

	kparams={
		'name': 'CommWordString',
		'args': {'key': ('use_sign',), 'val': (False,)},
		'data': dataop.get_dna(),
		'feature_class': 'string_complex',
		'feature_type': 'Word'
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=CommWordStringKernel(10, *kparams['args']['val'])
	output=fileop.get_output(category.KERNEL, kparams)
	_loop_svm(svms, params, feats, kernel, output)

	kparams={
		'name': 'CommUlongString',
		'args': {'key': ('use_sign',), 'val': (False,)},
		'data': dataop.get_dna(),
		'feature_class': 'string_complex',
		'feature_type': 'Ulong'
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=CommUlongStringKernel(10, *kparams['args']['val'])
	output=fileop.get_output(category.KERNEL, kparams)
	_loop_svm(svms, params, feats, kernel, output)

	kparams={
		'name': 'WeightedDegreeString',
		'args': {'key': ('degree',), 'val': (3,)},
		'data': dataop.get_dna(),
		'feature_class': 'string',
		'feature_type': 'Char'
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=WeightedDegreeStringKernel(*kparams['args']['val'])
	output=fileop.get_output(category.KERNEL, kparams)
	_loop_svm(svms, params, feats, kernel, output)
	params['linadd_enabled']=True
	_loop_svm(svms, params, feats, kernel, output)
	params['batch_enabled']=True
	_loop_svm(svms, params, feats, kernel, output)

	kparams={
		'name': 'WeightedDegreePositionString',
		'args': {'key': ('degree',), 'val': (20,)},
		'data': dataop.get_dna(),
		'feature_class': 'string',
		'feature_type': 'Char'
	}
	feats=featop.get_features(
		kparams['feature_class'], kparams['feature_type'], kparams['data'])
	kernel=WeightedDegreePositionStringKernel(10, *kparams['args']['val'])
	output=fileop.get_output(category.KERNEL, kparams)
	del params['linadd_enabled']
	del params['batch_enabled']
	_loop_svm(svms, params, feats, kernel, output)
	params['linadd_enabled']=True
	_loop_svm(svms, params, feats, kernel, output)
	params['batch_enabled']=True
	_loop_svm(svms, params, feats, kernel, output)


def _run_svm_linear ():
	"""Run all SVMs based on (Sparse) Linear Classifiers."""

	params={
		'type': 'linear',
		'bias_enabled': False,
		'data': dataop.get_clouds(2),
		'feature_class': 'simple',
		'feature_type': 'Real',
		'label_type': 'twoclass'
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'],
		params['data'], sparse=True)

	svms=('LibLinear', 'SVMLin', 'SVMSGD')
	params['bias_enabled']=True
	_loop_svm(svms, params, feats)

	# SubGradientSVM needs max_train_time to terminate
	svms=('SubGradientSVM',)
	params['bias_enabled']=False
	params['max_train_time']=.5 # up to 2. does not improve test results :(
	_loop_svm(svms, params, feats)

	svms=('SVMOcas',)
	_loop_svm(svms, params, feats)

	params={
		'type': 'linear',
		'bias_enabled': False,
		'label_type': 'twoclass',
		'feature_class': 'wd',
		'feature_type': 'Byte',
		'data': dataop.get_dna(),
		'alphabet': 'RAWDNA',
		'order': 1
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'],
		params['data'], params['order'])
	_loop_svm(svms, params, feats)



##########################################################################
# other classifiers
##########################################################################

def _run_perceptron ():
	"""Run Perceptron classifier."""

	params={
		'name': 'Perceptron',
		'type': 'perceptron',
		'num_threads': 1,
		'learn_rate': .1,
		'max_iter': 1000,
		'data': dataop.get_clouds(2),
		'feature_class': 'simple',
		'feature_type': 'Real',
		'label_type': 'twoclass',
		'accuracy': 1e-7
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	num_vec=feats['train'].get_num_vectors()
	params['labels'], labels=dataop.get_labels(num_vec, params['label_type'])

	perceptron=classifier.Perceptron(feats['train'], labels)
	perceptron.parallel.set_num_threads(params['num_threads'])
	perceptron.set_learn_rate(params['learn_rate'])
	perceptron.set_max_iter(params['max_iter'])
	perceptron.train()

	params['bias']=perceptron.get_bias()
	perceptron.set_features(feats['test'])
	params['classified']=perceptron.classify().get_labels()

	output=fileop.get_output(category.CLASSIFIER, params)
	fileop.write(category.CLASSIFIER, output)


def _run_knn ():
	"""Run K-Nearest-Neighbour classifier.
	"""

	params={
		'name': 'EuclidianDistance',
		'data': dataop.get_clouds(2),
		'feature_class': 'simple',
		'feature_type': 'Real'
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	dfun=eval(params['name'])
	distance=dfun(feats['train'], feats['train'])
	output=fileop.get_output(category.DISTANCE, params)

	params={
		'name': 'KNN',
		'type': 'knn',
		'num_threads': 1,
		'k': 3,
		'label_type': 'twoclass',
		'accuracy': 1e-8
	}
	params['labels'], labels=dataop.get_labels(
		feats['train'].get_num_vectors(), params['label_type'])

	knn=classifier.KNN(params['k'], distance, labels)
	knn.parallel.set_num_threads(params['num_threads'])
	knn.train()

	distance.init(feats['train'], feats['test'])
	params['classified']=knn.classify().get_labels()

	output.update(fileop.get_output(category.CLASSIFIER, params))
	fileop.write(category.CLASSIFIER, output)


def _run_lda ():
	"""Run Linear Discriminant Analysis classifier."""

	params={
		'name': 'LDA',
		'type': 'lda',
		'gamma': 0.1,
		'num_threads': 1,
		'data': dataop.get_clouds(2),
		'feature_class': 'simple',
		'feature_type': 'Real',
		'label_type': 'twoclass',
		'accuracy': 1e-7
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	params['labels'], labels=dataop.get_labels(
		feats['train'].get_num_vectors(), params['label_type'])

	lda=classifier.LDA(params['gamma'], feats['train'], labels)
	lda.parallel.set_num_threads(params['num_threads'])
	lda.train()

	lda.set_features(feats['test'])
	params['classified']=lda.classify().get_labels()

	output=fileop.get_output(category.CLASSIFIER, params)
	fileop.write(category.CLASSIFIER, output)


def _run_wdsvmocas ():
	"""Run Weighted Degree SVM Ocas classifier."""

	svms=('WDSVMOcas',)
	params={
		'type': 'wdsvmocas',
		'degree': 1,
		'bias_enabled': False,
		#'data': dataop.get_rawdna(),
		'data': dataop.get_dna(
			dataop.NUM_VEC_TRAIN, dataop.NUM_VEC_TRAIN, dataop.NUM_VEC_TRAIN),
		'feature_class': 'string_complex',
		'feature_type': 'Byte',
		'alphabet': 'RAWDNA',
		'label_type': 'twoclass',
		'order': 1,
		'gap': 0,
		'reverse': False
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'],
		params['data'], eval(params['alphabet']),
		params['order'], params['gap'], params['reverse'])
	_loop_svm(svms, params, feats)


##########################################################################
# public
##########################################################################

def run ():
	"""Run generator for all classifiers."""

#	_run_svm_kernel()
	_run_svm_linear()
#	_run_knn()
#	_run_lda()
#	_run_perceptron()
#	_run_wdsvmocas()

