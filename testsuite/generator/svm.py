from numpy import matrix
from numpy.random import rand
from shogun.Kernel import GaussianKernel
from shogun.Features import Labels
from shogun.Classifier import *

import fileop
import featop
import dataop
#from svmlist import SVMLIST

def _compute (name, feats, data, params):
	kargs=[1.5]
	k=GaussianKernel(feats['train'], feats['train'], 1.5)
	k.parallel.set_num_threads(params['num_threads'])

	num_vec=feats['train'].get_num_vectors();
	labels=rand(num_vec).round()*2-1
	l=Labels(labels)
	svmfun=eval(name)
	svm=svmfun(params['C'], k, l)
	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])
	svm.set_tube_epsilon(params['tube_epsilon'])
	svm.train()
	alphas=svm.get_alphas()
	bias=svm.get_bias()
	support_vectors=svm.get_support_vectors()

	k.init(feats['train'], feats['test'])
	classified=svm.classify().get_labels()

	output={
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'kname':'Gaussian',
		'svmparam_C':params['C'],
		'svmparam_epsilon':params['epsilon'],
		'svmparam_tube_epsilon':params['tube_epsilon'],
		'svmparam_num_threads':params['num_threads'],
		'svmparam_alphas':alphas,
		'svmparam_labels':labels,
		'svmparam_bias':bias,
		'svmparam_support_vectors':support_vectors,
		'svmparam_classified':classified
	}
	output.update(fileop.get_output_params('Gaussian', kargs))

	return [name, output]


def run ():
	fileop.TYPE='SVM'

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	params_svm={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2, 'num_threads':1}

	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['C']=.23
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['C']=1.5
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['C']=30
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['epsilon']=1e-4
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['tube_epsilon']=1e-3
	fileop.write(_compute('SVMLight', feats, data, params_svm))
	params_svm['num_threads']=16
	fileop.write(_compute('SVMLight', feats, data, params_svm))

