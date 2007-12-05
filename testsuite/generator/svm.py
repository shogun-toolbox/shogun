from numpy import matrix, double
from numpy.random import rand
from shogun.Kernel import *
from shogun.Features import Labels
from shogun.Classifier import *
from shogun.Library import E_WD

import fileop
import featop
import dataop
from svmlist import SVMLIST

LABEL_RANDOM=0
LABEL_SERIES=1

def _get_output_params(name, kernel, params):
	output={
		'data_train':matrix(kernel['data']['train']),
		'data_test':matrix(kernel['data']['test']),
		'kname':kernel['name'],
		'svmparam_C':params['C'],
		'svmparam_epsilon':params['epsilon'],
		'svmparam_tube_epsilon':params['tube_epsilon'],
		'svmparam_num_threads':params['num_threads'],
		'svmparam_alphas':params['alphas'],
		'svmparam_labels':params['labels'],
		'svmparam_bias':params['bias'],
		'svmparam_support_vectors':params['support_vectors'],
		'svmparam_classified':params['classified'],
		'svmparam_accuracy':SVMLIST[name][0],
	}
	output.update(fileop.get_output_params(kernel['name'], kernel['args']))

	return output

def _compute (name, kernel, params):
	kernel['k'].parallel.set_num_threads(params['num_threads'])
	kernel['k'].init(kernel['feats']['train'], kernel['feats']['train'])

	svmfun=eval(name)

	if params['labels'] is None:
		svm=svmfun(params['C'], kernel['k'])
	else:
		l=Labels(params['labels'])
		svm=svmfun(params['C'], kernel['k'], l)

	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])
	svm.set_tube_epsilon(params['tube_epsilon'])
	svm.train()
	params['alphas']=svm.get_alphas()
	params['bias']=svm.get_bias()
	params['support_vectors']=svm.get_support_vectors()

	kernel['k'].init(kernel['feats']['train'], kernel['feats']['test'])
	params['classified']=svm.classify().get_labels()

	return [name, _get_output_params(name, kernel, params)]

def _run (svms, kernel, labels=LABEL_RANDOM):
	kfun=eval(kernel['name']+'Kernel')
	# FIXME: cache size has to go....
	kernel['k']=kfun(10, *kernel['args'])
	# FIXME: NASTY NASTY NASTY! but WeightedStringKernel is a bit inconsistent
	# in constructors, so have to get rid of first arg EWDKernType
	if kernel['name']=='WeightedDegreeString':
		kernel['args']=kernel['args'][1:]

	num_vec=kernel['feats']['train'].get_num_vectors();
	if labels==LABEL_RANDOM:
		labels=rand(num_vec).round()*2-1
	elif labels==LABEL_SERIES:
		labels=[double(x) for x in xrange(num_vec)]

	for name in svms:
		params={
			'C':.017,
			'epsilon':1e-5,
			'tube_epsilon':1e-2,
			'num_threads':1,
			'labels':labels
		}
		fileop.write(_compute(name, kernel, params))
		params['C']=.23
		fileop.write(_compute(name, kernel,  params))
		params['C']=1.5
		fileop.write(_compute(name, kernel, params))
		params['C']=30
		fileop.write(_compute(name, kernel, params))
		params['epsilon']=1e-4
		fileop.write(_compute(name, kernel, params))
		params['tube_epsilon']=1e-3
		fileop.write(_compute(name, kernel, params))
		params['num_threads']=16
		fileop.write(_compute(name, kernel, params))

def _run_feats_real ():
	svms=['SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM']
	kernel={
		'name':'Gaussian',
		'data':dataop.get_rand(),
		'args':[1.5]
	}
	kernel['feats']=featop.get_simple('Real', kernel['data'])
	_run(svms, kernel)

	svms=['LibSVMMultiClass', 'GMNPSVM']
	_run(svms, kernel, LABEL_SERIES)

#	svms=['LibSVMOneClass']
#	_run(svms, kernel, None)

	svms=['SVMLight', 'GPBTSVM']
	kernel['name']='Linear'
	_run(svms, kernel)

def _run_feats_string ():
	svms=['SVMLight', 'GPBTSVM']
	kernel={
		'data':dataop.get_dna(),
	}
	kernel['feats']=featop.get_string('Char', kernel['data'])

	kernel['name']='WeightedDegreeString'
	kernel['args']=[E_WD, 3, 0]
	_run(svms, kernel)

	kernel['name']='WeightedDegreePositionString'
	kernel['args']=[20]
	_run(svms, kernel)


def _run_feats_string_complex ():
	svms=['SVMLight', 'GPBTSVM']
	kernel={
		'data':dataop.get_dna(),
		'args':[False, FULL_NORMALIZATION]
	}

	kernel['name']='CommWordString'
	kernel['feats']=featop.get_string_complex('Word', kernel['data'])
	_run(svms, kernel)

	kernel['name']='CommUlongString'
	kernel['feats']=featop.get_string_complex('Ulong', kernel['data'])
	_run(svms, kernel)

def run ():
	fileop.TYPE='SVM'

	_run_feats_real()
	_run_feats_string()
	_run_feats_string_complex()



