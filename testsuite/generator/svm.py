from numpy import matrix
from numpy.random import rand
from shogun.Kernel import GaussianKernel
from shogun.Features import Labels
from shogun.Classifier import *

import fileop
import featop
import dataop
from svmlist import SVMLIST

def _get_output_params(name, data, params, kargs):
	output={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'kname':'Gaussian',
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
	output.update(fileop.get_output_params('Gaussian', kargs))

	return output

def _compute (name, feats, data, params):
	kargs=[1.5]
	k=GaussianKernel(feats['train'], feats['train'], *kargs)
	k.parallel.set_num_threads(params['num_threads'])

	num_vec=feats['train'].get_num_vectors();
	params['labels']=rand(num_vec).round()*2-1
	l=Labels(params['labels'])
	svmfun=eval(name)
	svm=svmfun(params['C'], k, l)
	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])
	svm.set_tube_epsilon(params['tube_epsilon'])
	svm.train()
	params['alphas']=svm.get_alphas()
	params['bias']=svm.get_bias()
	params['support_vectors']=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	params['classified']=svm.classify().get_labels()

	return [name, _get_output_params(name, data, params, kargs)]

def _run (feats, data, params_svm):
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	fileop.write(_compute('LibSVM', feats, data, params_svm))
	fileop.write(_compute('GPBTSVM', feats, data, params_svm))
	fileop.write(_compute('MPDSVM', feats, data, params_svm))

def run ():
	fileop.TYPE='SVM'

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	params_svm={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2, 'num_threads':1}

	_run(feats, data, params_svm)
	params_svm['C']=.23
	_run(feats, data, params_svm)
	params_svm['C']=1.5
	_run(feats, data, params_svm)
	params_svm['C']=30
	_run(feats, data, params_svm)
	params_svm['epsilon']=1e-4
	_run(feats, data, params_svm)
	params_svm['tube_epsilon']=1e-3
	_run(feats, data, params_svm)
	params_svm['num_threads']=16
	_run(feats, data, params_svm)

