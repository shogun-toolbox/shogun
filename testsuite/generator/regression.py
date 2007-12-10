from numpy import *
from numpy.random import rand
from shogun.Kernel import GaussianKernel
from shogun.Features import Labels
from shogun.Regression import *

import fileop
import featop
import dataop
from regressionlist import REGRESSIONLIST


def _get_output_params (name, params, data):
	type=REGRESSIONLIST[name][1]
	output={
		'name':name,
		'data_train':matrix(data['data']['train']),
		'data_test':matrix(data['data']['test']),
		'regression_accuracy':REGRESSIONLIST[name][0],
		'regression_type':type,
	}

	for k, v in params.iteritems():
		output['regression_'+k]=v

	output['kernel_name']=data['kname']
	kparams=fileop.get_output_params(
		data['kname'], fileop.T_KERNEL, data['kargs'])
	output.update(kparams)

	return output

def _compute (name, params, data):
	type=REGRESSIONLIST[name][1]
	data['k'].parallel.set_num_threads(params['num_threads'])
	data['k'].init(data['feats']['train'], data['feats']['train'])
	# essential to wrap in array(), will segfault sometimes otherwise
	labels=Labels(array(params['labels']))

	fun=eval(name)
	if type=='svm':
		regression=fun(params['C'], params['epsilon'], data['k'], labels)
	else:
		regression=fun(params['tau'], data['k'], labels)
	regression.parallel.set_num_threads(params['num_threads'])

	if params.has_key('tube_epsilon'):
		regression.set_tube_epsilon(params['tube_epsilon'])

	regression.train()

	if type=='svm':
		params['bias']=regression.get_bias()
		params['alphas']=regression.get_alphas()
		params['support_vectors']=regression.get_support_vectors()

	data['k'].init(data['feats']['train'], data['feats']['test'])
	params['classified']=regression.classify().get_labels()

	output=_get_output_params(name, params, data)
	fileop.write(fileop.T_REGRESSION, output)

def _loop (svrs, data):
	num_vec=data['feats']['train'].get_num_vectors()
	labels=rand(num_vec).round()*2-1
	for name in svrs:
		type=REGRESSIONLIST[name][1]
		if type=='svm':
			params={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2,
				'num_threads':1, 'labels':labels}
		elif type=='kernelmachine':
			params={'tau':1e-6, 'num_threads':1, 'labels':labels}
		else:
			continue

		_compute(name, params, data)

		if type=='svm':
			params['C']=.23
			_compute(name, params, data)
			params['C']=1.5
			_compute(name, params, data)
			params['C']=30
			_compute(name, params, data)
			params['epsilon']=1e-4
			_compute(name, params, data)
			params['tube_epsilon']=1e-3
			_compute(name, params, data)
		elif type=='kernelmachine':
			params['tau']=1e-5
			_compute(name, params, data)
		else:
			continue

		# BUG in SVRLight:
		# glibc detected *** /usr/bin/python: free(): invalid next size (fast)
		if name!='SVRLight':
			params['num_threads']=16
			_compute(name, params, data)

##########################################################################
# public
##########################################################################

def run ():
	svrs=['SVRLight', 'LibSVR', 'KRR']
	data={
		'kname':'Gaussian',
		'kargs':[1.5],
		'data':dataop.get_rand(),
	}
	data['feats']=featop.get_simple('Real', data['data'])
	data['k']=GaussianKernel(10, *data['kargs'])
	_loop(svrs, data)

