"""Generator for Classifier"""

from numpy import *
from numpy.random import rand
from shogun.Kernel import *
from shogun.Distance import EuclidianDistance
from shogun.Features import Labels
from shogun.Classifier import *
from shogun.Library import E_WD

import fileop
import featop
import dataop
from config import CLASSIFIER, C_KERNEL, C_DISTANCE, C_CLASSIFIER

def _get_outdata (name, params):
	"""Return data to be written into the testcase's file.

	After computations and such, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.
	
	@param name Classifier's name
	@param params Gathered data
	@return Dict containing testcase data to be written to file
	"""

	ctype=CLASSIFIER[name][1]
	outdata={
		'name':name,
		'data_train':matrix(params['data']['train']),
		'data_test':matrix(params['data']['test']),
		'classifier_accuracy':CLASSIFIER[name][0],
		'classifier_type':ctype,
	}

	optional=['num_threads', 'classified',
		'alphas', 'bias','support_vectors', 'labels',
		'C', 'epsilon',
		'bias_enabled',
		'tube_epsilon',
		'max_train_time',
		'max_iter', 'learn_rate',
		'k',
		'gamma',
	]
	for opt in optional:
		if params.has_key(opt):
			outdata['classifier_'+opt]=params[opt]

	if ctype=='kernel':
		outdata['kernel_name']=params['kname']
		kparams=fileop.get_outdata(params['kname'], C_KERNEL, params['kargs'])
		outdata.update(kparams)
	elif ctype=='knn':
		outdata['distance_name']=params['dname']
		dparams=fileop.get_outdata(
			params['dname'], C_DISTANCE, params['dargs'])
		outdata.update(dparams)
	else:
		outdata['feature_class']='simple'
		outdata['feature_type']='Real'
		outdata['data_type']='double'
	return outdata

##########################################################################
# svm
##########################################################################

def _get_svm (name, labels, params):
	"""Return an SVM object.

	This function instantiates an SVM depending on the parameters.

	@param name Name of the SVM to instantiate
	@param labels Labels to be used for the SVM (if at all!)
	@param params Misc parameters for the SVM's constructor
	@return An SVM object
	"""

	svmfun=eval(name)
	ctype=CLASSIFIER[name][1]

	if ctype=='kernel':
		params['kernel'].parallel.set_num_threads(params['num_threads'])
		params['kernel'].init(
			params['feats']['train'], params['feats']['train'])

		if labels is None:
			return svmfun(params['C'], params['kernel'])
		else:
			return svmfun(params['C'], params['kernel'], labels)
	else:
		return svmfun(params['C'], params['feats']['train'], labels)

def _compute_svm (name, labels, params):
	"""Perform computations on SVM.

	Perform all necessary computations on SVM and gather the output.

	@param name Name of the SVM to instantiate
	@param labels Labels to be used for the SVM (if at all!)
	@param params Misc parameters for the SVM's constructor
	"""

	ctype=CLASSIFIER[name][1]
	svm=_get_svm(name, labels, params)
	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])

	if params.has_key('tube_epsilon'):
		svm.set_tube_epsilon(params['tube_epsilon'])
	if params.has_key('bias_enabled'):
		svm.set_bias_enabled(params['bias_enabled'])
	if params.has_key('max_train_time'):
		svm.set_max_train_time(params['max_train_time'])
		params['max_train_time']=params['max_train_time']

	svm.train()

	if ((params.has_key('bias_enabled') and params['bias_enabled']) or
		ctype=='kernel'):
		params['bias']=svm.get_bias()

	if ctype=='kernel':
		alphas=svm.get_alphas()
		if len(alphas)>0:
			params['alphas']=alphas

		support_vectors=svm.get_support_vectors()
		if len(support_vectors)>0:
			params['support_vectors']=support_vectors

		params['kernel'].init(
			params['feats']['train'], params['feats']['test'])
	elif ctype=='linear':
		svm.set_features(params['feats']['test'])

	params['classified']=svm.classify().get_labels()

	outdata=_get_outdata(name, params)
	fileop.write(C_CLASSIFIER, outdata)

def _loop_svm (svms, params):
	"""Loop through SVM computations, only slightly differing in parameters.

	Loop through SVM computations with little variations in the parameters for the SVM. Not necessarily used by all SVMs in this generator.

	@param svms Names of the svms to loop through
	@param params Parameters to the SVM
	"""

	for name in svms:
		ctype=CLASSIFIER[name][1]
		ltype=CLASSIFIER[name][2]

		parms={
			'num_threads':1,
			'C':.017,
			'epsilon':1e-5,
		}
		parms.update(params)

		if ctype=='kernel':
			parms['tube_epsilon']=1e-2

		if ltype is not None:
			parms['labels'], labels=dataop.get_labels(
				params['feats']['train'].get_num_vectors(), ltype)
		else:
			labels=None

		_compute_svm(name, labels, parms)
		parms['C']=.23
		_compute_svm(name, labels, parms)
		parms['C']=1.5
		_compute_svm(name, labels, parms)
		parms['C']=30
		_compute_svm(name, labels, parms)
		parms['epsilon']=1e-4
		_compute_svm(name, labels, parms)

		if ctype=='kernel':
			parms['tube_epsilon']=1e-3
			_compute_svm(name, labels, parms)

		parms['num_threads']=16
		_compute_svm(name, labels, parms)

def _run_svm_kernel ():
	"""Run all kernel-based SVMs."""

	svms=['SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM', 'LibSVMOneClass']
	params={
		'kname':'Gaussian',
		'kargs':[1.5],
	}
	params['data']=dataop.get_clouds(2)
	params['feats']=featop.get_simple('Real', params['data'])
	params['kernel']=GaussianKernel(10, *params['kargs'])
	_loop_svm(svms, params)

	svms=['LibSVMMultiClass', 'GMNPSVM']
	params['data']=dataop.get_clouds(3)
	params['feats']=featop.get_simple('Real', params['data'])
	_loop_svm(svms, params)

	svms=['SVMLight', 'GPBTSVM']
	params['kname']='Linear'
	params['kernel']=LinearKernel(10, *params['kargs'])
	_loop_svm(svms, params)

	params['data']=dataop.get_dna()
	params['feats']=featop.get_string('Char', params['data'])
	params['kname']='WeightedDegreeString'
	params['kargs']=[3]
	params['kernel']=WeightedDegreeStringKernel(*params['kargs'])
	_loop_svm(svms, params)

	params['kname']='WeightedDegreePositionString'
	params['kargs']=[20]
	params['kernel']=WeightedDegreePositionStringKernel(10, *params['kargs'])
	_loop_svm(svms, params)

	params['kargs']=[False, FULL_NORMALIZATION]
	params['kname']='CommWordString'
	params['feats']=featop.get_string_complex('Word', params['data'])
	params['kernel']=CommWordStringKernel(10, *params['kargs'])
	_loop_svm(svms, params)

	params['kname']='CommUlongString'
	params['feats']=featop.get_string_complex('Ulong', params['data'])
	params['kernel']=CommUlongStringKernel(10, *params['kargs'])
	_loop_svm(svms, params)

def _run_svm_linear ():
	"""Run all SVMs based on (Sparse) Linear Classifiers."""

	svms=['SVMOcas']
	params={
		'data':dataop.get_clouds(2),
		'bias_enabled':False,
	}
	params['feats']=featop.get_simple('Real', params['data'], sparse=True)
	_loop_svm(svms, params)

	svms=['LibLinear', 'SVMLin']
	params['bias_enabled']=True
	_loop_svm(svms, params)

	# SubGradientSVM needs max_train_time to terminate
	svms=['SubGradientSVM']
	params['bias_enabled']=False
	params['max_train_time']=.5 # up to 2. does not improve test results :(
	_loop_svm(svms, params)

##########################################################################
# other classifiers
##########################################################################

def _run_perceptron ():
	"""Run Perceptron classifier."""

	name='Perceptron'
	params={
		'num_threads':1,
		'learn_rate':.1,
		'max_iter':1000,
		'data':dataop.get_clouds(2)
	}
	feats=featop.get_simple('Real', params['data'])
	num_vec=feats['train'].get_num_vectors()
	params['labels'], labels=dataop.get_labels(num_vec, CLASSIFIER[name][2])

	perceptron=Perceptron(feats['train'], labels)
	perceptron.parallel.set_num_threads(params['num_threads'])
	perceptron.set_learn_rate(params['learn_rate'])
	perceptron.set_max_iter(params['max_iter'])
	perceptron.train()

	params['bias']=perceptron.get_bias()
	perceptron.set_features(feats['test'])
	params['classified']=perceptron.classify().get_labels()

	outdata=_get_outdata(name, params)
	fileop.write(C_CLASSIFIER, outdata)

def _run_knn ():
	"""Run K-Nearest-Neighbour classifier.
	"""

	name='KNN'
	params={
		'num_threads':1,
		'k':3,
		'dname':'EuclidianDistance',
		'dargs':[],
		'data':dataop.get_clouds(2),
	}
	feats=featop.get_simple('Real', params['data'])
	fun=eval(params['dname'])
	distance=fun(feats['train'], feats['train'], *params['dargs'])
	params['labels'], labels=dataop.get_labels(
		feats['train'].get_num_vectors(), CLASSIFIER[name][2])

	knn=KNN(params['k'], distance, labels)
	knn.parallel.set_num_threads(params['num_threads'])
	knn.train()

	distance.init(feats['train'], feats['test'])
	params['classified']=knn.classify().get_labels()

	outdata=_get_outdata(name, params)
	fileop.write(C_CLASSIFIER, outdata)

def _run_lda ():
	"""Run Linear Discriminant Analysis classifier."""

	name='LDA'
	params={
		'gamma':.1,
		'num_threads':1,
		'data':dataop.get_clouds(2),
	}
	feats=featop.get_simple('Real', params['data'])
	params['labels'], labels=dataop.get_labels(
		feats['train'].get_num_vectors(), CLASSIFIER[name][2])

	lda=LDA(params['gamma'], feats['train'], labels)
	lda.parallel.set_num_threads(params['num_threads'])
	lda.train()

	lda.set_features(feats['test'])
	params['classified']=lda.classify().get_labels()

	outdata=_get_outdata(name, params)
	fileop.write(C_CLASSIFIER, outdata)

##########################################################################
# public
##########################################################################

def run ():
	"""Run generator for all classifiers."""

	_run_svm_kernel()
	_run_svm_linear()
	_run_knn()
	_run_lda()
	_run_perceptron()



