"""
Generator for Regression
"""

from numpy import *
from numpy.random import rand
from shogun.Kernel import GaussianKernel
from shogun.Regression import *

import fileop
import featop
import dataop
from config import REGRESSION, C_KERNEL, C_REGRESSION


def _get_outdata_params (name, params, data):
	rtype=REGRESSION[name][1]
	outdata={
		'name':name,
		'data_train':matrix(data['data']['train']),
		'data_test':matrix(data['data']['test']),
		'regression_accuracy':REGRESSION[name][0],
		'regression_type':rtype,
	}

	for key, val in params.iteritems():
		outdata['regression_'+key]=val

	outdata['kernel_name']=data['kname']
	kparams=fileop.get_outdata_params(
		data['kname'], C_KERNEL, data['kargs'])
	outdata.update(kparams)

	return outdata

def _compute (name, params, data):
	rtype=REGRESSION[name][1]
	data['kernel'].parallel.set_num_threads(params['num_threads'])
	data['kernel'].init(data['feats']['train'], data['feats']['train'])
	params['labels'], labels=dataop.get_labels(
		data['feats']['train'].get_num_vectors())

	fun=eval(name)
	if rtype=='svm':
		regression=fun(params['C'], params['epsilon'], data['kernel'], labels)
	else:
		regression=fun(params['tau'], data['kernel'], labels)
	regression.parallel.set_num_threads(params['num_threads'])

	if params.has_key('tube_epsilon'):
		regression.set_tube_epsilon(params['tube_epsilon'])

	regression.train()

	if rtype=='svm':
		params['bias']=regression.get_bias()
		params['alphas']=regression.get_alphas()
		params['support_vectors']=regression.get_support_vectors()

	data['kernel'].init(data['feats']['train'], data['feats']['test'])
	params['classified']=regression.classify().get_labels()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_REGRESSION, outdata)

def _loop (svrs, data):
	num_vec=data['feats']['train'].get_num_vectors()
	for name in svrs:
		rtype=REGRESSION[name][1]
		if rtype=='svm':
			params={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2,
				'num_threads':1}
		elif rtype=='kernelmachine':
			params={'tau':1e-6, 'num_threads':1}
		else:
			continue

		_compute(name, params, data)

		if rtype=='svm':
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
		elif rtype=='kernelmachine':
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
	data['kernel']=GaussianKernel(10, *data['kargs'])
	_loop(svrs, data)

