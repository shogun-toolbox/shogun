"""
Generator for Classifier

some words about params + data (also valid for other generators):
- params contains parameters to tweak a classifiers operation of which all
  are written to the outdata file.
- data contains all sorts of objects to setup the classifier of which some
  are written to the outdata file.

there might be a better solution...
"""

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

def _get_outdata_params (name, params, data):
	ctype=CLASSIFIER[name][1]
	outdata={
		'name':name,
		'data_train':matrix(data['data']['train']),
		'data_test':matrix(data['data']['test']),
		'classifier_accuracy':CLASSIFIER[name][0],
		'classifier_type':ctype,
	}

	for key, val in params.iteritems():
		outdata['classifier_'+key]=val

	if ctype=='kernel':
		outdata['kernel_name']=data['kname']
		kparams=fileop.get_outdata_params(
			data['kname'], C_KERNEL, data['kargs'])
		outdata.update(kparams)
	elif ctype=='knn':
		outdata['distance_name']=data['dname']
		dparams=fileop.get_outdata_params(
			data['dname'], C_DISTANCE, data['dargs'])
		outdata.update(dparams)
	else:
		outdata['feature_class']='simple'
		outdata['feature_type']='Real'
		outdata['data_type']='double'

	return outdata

def _get_labels (ltype, num):
	labels=[]
	if ltype=='twoclass':
		labels.append(rand(num).round()*2-1)
	elif ltype=='series':
		labels.append([double(x) for x in xrange(num)])
	else:
		return [None, None]

	# essential to wrap in array(), will segfault sometimes otherwise
	labels.append(Labels(array(labels[0])))

	return labels

##########################################################################
# svm
##########################################################################

def _get_svm (name, labels, params, data):
	svmfun=eval(name)
	ctype=CLASSIFIER[name][1]

	if ctype=='kernel':
		data['kernel'].parallel.set_num_threads(params['num_threads'])
		data['kernel'].init(data['feats']['train'], data['feats']['train'])
		if labels is None:
			return svmfun(params['C'], data['kernel'])
		else:
			return svmfun(params['C'], data['kernel'], labels)
	else:
		return svmfun(params['C'], data['feats']['train'], labels)

def _compute_svm (name, labels, params, data):
	ctype=CLASSIFIER[name][1]
	svm=_get_svm(name, labels, params, data)
	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])

	if params.has_key('tube_epsilon'):
		svm.set_tube_epsilon(params['tube_epsilon'])

	if data.has_key('bias_enabled'):
		svm.set_bias_enabled(data['bias_enabled'])

	svm.train()

	if data.has_key('bias_enabled') and data['bias_enabled']:
		params['bias']=svm.get_bias()
	else:
		if ctype=='kernel':
			params['bias']=svm.get_bias()

			alphas=svm.get_alphas()
			if len(alphas)>0:
				params['alphas']=alphas

			support_vectors=svm.get_support_vectors()
			if len(support_vectors)>0:
				params['support_vectors']=support_vectors

			data['kernel'].init(data['feats']['train'], data['feats']['test'])

	params['classified']=svm.classify().get_labels()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_CLASSIFIER, outdata)

def _loop_svm (svms, data):
	for name in svms:
		ctype=CLASSIFIER[name][1]
		ltype=CLASSIFIER[name][2]

		if ctype=='kernel':
			params={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2,
				'num_threads':1}
		else:
			params={'C':.017, 'epsilon':1e-5, 'num_threads':1}

		if ltype is not None:
			params['labels'], labels=_get_labels(
				ltype, data['feats']['train'].get_num_vectors())
		else:
			labels=None

		_compute_svm(name, labels, params, data)
		params['C']=.23
		_compute_svm(name, labels, params, data)
		params['C']=1.5
		_compute_svm(name, labels, params, data)
		params['C']=30
		_compute_svm(name, labels, params, data)
		params['epsilon']=1e-4
		_compute_svm(name, labels, params, data)

		if ctype=='kernel':
			params['tube_epsilon']=1e-3
			_compute_svm(name, labels, params, data)

		params['num_threads']=16
		_compute_svm(name, labels, params, data)

def _run_svm_kernel ():
	svms=['SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM', 'LibSVMMultiClass', 'GMNPSVM']
	#svms=['SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM', 'LibSVMMultiClass',
	#	'GMNPSVM', 'LibSVMOneClass']
	data={
		'kname':'Gaussian',
		'kargs':[1.5],
		'data':dataop.get_rand(),
	}
	data['feats']=featop.get_simple('Real', data['data'])
	data['kernel']=GaussianKernel(10, *data['kargs'])
	_loop_svm(svms, data)

	svms=['SVMLight', 'GPBTSVM']
	data['kname']='Linear'
	data['kernel']=LinearKernel(10, *data['kargs'])
	_loop_svm(svms, data)

	data['data']=dataop.get_dna()
	data['feats']=featop.get_string('Char', data['data'])
	data['kname']='WeightedDegreeString'
	data['kargs']=[E_WD, 3, 0]
	data['kernel']=WeightedDegreeStringKernel(10, *data['kargs'])
	# WeightedStringKernel is a bit inconsistent in constructors: have to get
	# rid of first arg EWDKernType in order to fit into scheme 
	data['kargs']=data['kargs'][1:]
	_loop_svm(svms, data)

	data['kname']='WeightedDegreePositionString'
	data['kargs']=[20]
	data['kernel']=WeightedDegreePositionStringKernel(10, *data['kargs'])
	_loop_svm(svms, data)

	data['kargs']=[False, FULL_NORMALIZATION]
	data['kname']='CommWordString'
	data['feats']=featop.get_string_complex('Word', data['data'])
	data['kernel']=CommWordStringKernel(10, *data['kargs'])
	_loop_svm(svms, data)

	data['kname']='CommUlongString'
	data['feats']=featop.get_string_complex('Ulong', data['data'])
	data['kernel']=CommUlongStringKernel(10, *data['kargs'])
	_loop_svm(svms, data)

def _run_svm_linear ():
	#svms=['SubGradientSVM', 'SVMOcas']
	svms=['SVMOcas']
	data={
		'data':dataop.get_rand(),
		'bias_enabled':False,
	}
	data['feats']=featop.get_simple('Real', data['data'], sparse=True)
	_loop_svm(svms, data)

	svms=['LibLinear', 'SVMLin']
	data['bias_enabled']=True
	_loop_svm(svms, data)


##########################################################################
# other classifiers
##########################################################################

def _run_perceptron ():
	name='Perceptron'
	params={
		'num_threads':1,
		'learn_rate':.1,
		'max_iter':1000,
	}
	data={'data':dataop.get_rand()}
	feats=featop.get_simple('Real', data['data'])
	num_vec=feats['train'].get_num_vectors()
	params['labels'], labels=_get_labels(CLASSIFIER[name][2], num_vec)
	weights=rand(num_vec)

	perceptron=Perceptron(feats['train'], labels)
	perceptron.parallel.set_num_threads(params['num_threads'])
	perceptron.set_learn_rate(params['learn_rate'])
	perceptron.set_max_iter(params['max_iter'])
	perceptron.set_w(weights, num_vec)
	perceptron.train()

	params['bias']=perceptron.get_bias()
	params['classified']=perceptron.classify().get_labels()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_CLASSIFIER, outdata)

def _run_knn ():
	name='KNN'
	params={
		'num_threads':1,
		'k':3,
	}
	data={
		'dname':'EuclidianDistance',
		'dargs':[],
		'data':dataop.get_rand(),
	}
	feats=featop.get_simple('Real', data['data'])
	fun=eval(data['dname'])
	distance=fun(feats['train'], feats['train'], *data['dargs'])
	params['labels'], labels=_get_labels(
		CLASSIFIER[name][2], feats['train'].get_num_vectors())

	knn=KNN(params['k'], distance, labels)
	knn.parallel.set_num_threads(params['num_threads'])
	knn.train()

	distance.init(feats['train'], feats['test'])
	params['classified']=knn.classify().get_labels()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_CLASSIFIER, outdata)

def _run_lda ():
	name='LDA'

	params={
		'gamma':.1,
		'num_threads':1,
	}
	data={
		'data':dataop.get_rand(),
	}
	feats=featop.get_simple('Real', data['data'])
	params['labels'], labels=_get_labels(
		CLASSIFIER[name][2], feats['train'].get_num_vectors())

	lda=LDA(params['gamma'], feats['train'], labels)
	lda.parallel.set_num_threads(params['num_threads'])
	lda.train()

	params['classified']=lda.classify().get_labels()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_CLASSIFIER, outdata)

##########################################################################
# public
##########################################################################

def run ():
	#_run_svm_kernel()
	#_run_svm_linear()
	_run_knn()
	_run_lda()
	#_run_perceptron()



