"""Generator for Regression"""

import numpy
import shogun.Regression as regression
from shogun.Kernel import GaussianKernel

import fileop
import featop
import dataop
import config


def _get_outdata (name, params):
	"""Return data to be written into the testcase's file.

	After computations and such, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.
	
	@param name Regression method's name
	@param params Gathered data
	@return Dict containing testcase data to be written to file
	"""

	rtype=config.REGRESSION[name][1]
	outdata={
		'name':name,
		'data_train':numpy.matrix(params['data']['train']),
		'data_test':numpy.matrix(params['data']['test']),
		'init_random':dataop.INIT_RANDOM,
		'regression_accuracy':config.REGRESSION[name][0],
		'regression_type':rtype,
	}

	optional=['labels', 'num_threads', 'classified',
		'bias', 'alphas', 'support_vectors', 'C', 'epsilon', 'tube_epsilon',
		'tau',
	]
	for opt in optional:
		if params.has_key(opt):
			outdata['regression_'+opt]=params[opt]

	outdata['kernel_name']=params['kname']
	kparams=fileop.get_outdata(params['kname'], config.C_KERNEL,
		params['kargs'])
	outdata.update(kparams)

	return outdata

def _compute (name, params):
	"""Compute a regression and gather result data.

	@param name Name of the regression method
	@param params Misc parameters for the regression method's constructor
	"""

	rtype=config.REGRESSION[name][1]
	params['kernel'].parallel.set_num_threads(params['num_threads'])
	params['kernel'].init(params['feats']['train'], params['feats']['train'])
	params['labels'], labels=dataop.get_labels(
		params['feats']['train'].get_num_vectors())

	fun=eval('regression.'+name)
	if rtype=='svm':
		regression=fun(params['C'], params['epsilon'], params['kernel'], labels)
		regression.set_tube_epsilon(params['tube_epsilon'])
	else:
		regression=fun(params['tau'], params['kernel'], labels)
	regression.parallel.set_num_threads(params['num_threads'])

	regression.train()

	if rtype=='svm':
		params['bias']=regression.get_bias()
		params['alphas']=regression.get_alphas()
		params['support_vectors']=regression.get_support_vectors()

	params['kernel'].init(params['feats']['train'], params['feats']['test'])
	params['classified']=regression.classify().get_labels()

	outdata=_get_outdata(name, params)
	fileop.write(config.C_REGRESSION, outdata)

def _loop (svrs, params):
	"""Loop through regression computations, only slightly differing in parameters.
	@param svrs Names of the regression methods to loop through
	@param params Parameters of the regression method
	"""

	for name in svrs:
		rtype=config.REGRESSION[name][1]
		parms={'num_threads':1}
		parms.update(params)
		if rtype=='svm':
			parms['C']=.017
			parms['epsilon']=1e-5
			parms['tube_epsilon']=1e-2
			_compute(name, parms)
			parms['C']=.23
			_compute(name, parms)
			parms['C']=1.5
			_compute(name, parms)
			parms['C']=30
			_compute(name, parms)
			parms['epsilon']=1e-4
			_compute(name, parms)
			parms['tube_epsilon']=1e-3
			_compute(name, parms)
		elif rtype=='kernelmachine':
			parms['tau']=1e-6
			_compute(name, parms)
			parms['tau']=1e-5
			_compute(name, parms)
		else:
			continue

		# BUG in SVRLight:
		# glibc detected *** /usr/bin/python: free(): invalid next size (fast)
		if name!='SVRLight':
			parms['num_threads']=16
			_compute(name, parms)

##########################################################################
# public
##########################################################################

def run ():
	"""Run generator for all regression methods."""

	svrs=['SVRLight', 'LibSVR', 'KRR']
	params={
		'kname':'Gaussian',
		'kargs':[1.5],
		'data':dataop.get_rand(),
	}
	params['feats']=featop.get_simple('Real', params['data'])
	params['kernel']=GaussianKernel(10, *params['kargs'])
	_loop(svrs, params)

