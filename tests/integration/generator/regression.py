"""Generator for Regression"""

import shogun.Regression as regression
from shogun.Kernel import GaussianKernel

import fileop
import featop
import dataop
import category


def _compute (params, feats, kernel, pout):
	"""
	Compute a regression and gather result data.

	@param params misc parameters for the regression method
	@param feats features of the kernel/regression
	@param kernel kernel
	@param pout previously gathered data from kernel ready to be written to file
	"""

	kernel.parallel.set_num_threads(params['num_threads'])
	kernel.init(feats['train'], feats['train'])
	params['labels'], labels=dataop.get_labels(feats['train'].get_num_vectors())

	try:
		fun=eval('regression.'+params['name'])
	except AttributeError:
		return

	if params['type']=='svm':
		regression=fun(params['C'], params['epsilon'], kernel, labels)
		regression.set_tube_epsilon(params['tube_epsilon'])
	else:
		regression=fun(params['tau'], kernel, labels)
	regression.parallel.set_num_threads(params['num_threads'])

	regression.train()

	if params['type']=='svm':
		params['bias']=regression.get_bias()
		params['alpha_sum']=0
		for item in regression.get_alphas().tolist():
			params['alpha_sum']+=item
		params['sv_sum']=0
		for item in regression.get_support_vectors():
			params['sv_sum']+=item

	kernel.init(feats['train'], feats['test'])
	params['classified']=regression.apply().get_labels()

	output=pout.copy()
	output.update(fileop.get_output(category.REGRESSION, params))
	fileop.write(category.REGRESSION, output)


def _loop (regressions, feats, kernel, pout):
	"""
	Loop through regression computations, only slightly differing in parameters.

	@param regressions names of the regression methods to loop through
	@param feats features of the kernel/regression
	@param kernel kernel
	@param pout previously gathered data from kernel ready to be written to file
	"""

	for r in regressions:
		r['num_threads']=1
		if r['type']=='svm':
			r['C']=.017
			r['epsilon']=1e-5
			r['accuracy']=r['epsilon']*10
			r['tube_epsilon']=1e-2
			_compute(r, feats, kernel, pout)
			r['C']=.23
			_compute(r, feats, kernel, pout)
			r['C']=1.5
			_compute(r, feats, kernel, pout)
			r['C']=30
			_compute(r, feats, kernel, pout)
			r['epsilon']=1e-4
			r['accuracy']=r['epsilon']*10
			_compute(r, feats, kernel, pout)
			r['tube_epsilon']=1e-3
			_compute(r, feats, kernel, pout)
		elif r['type']=='kernelmachine':
			r['tau']=1e-6
			_compute(r, feats, kernel, pout)
			r['tau']=1e-5
			_compute(r, feats, kernel, pout)
		else:
			continue

		# BUG in SVRLight:
		# glibc detected *** /usr/bin/python: free(): invalid next size (fast)
		if r['name']!='SVRLight':
			r['num_threads']=16
			_compute(r, feats, kernel, pout)

##########################################################################
# public
##########################################################################

def run ():
	"""Run generator for all regression methods."""

	regressions=(
		{'name': 'SVRLight', 'type': 'svm', 'accuracy': 1e-6},
		{'name': 'LibSVR', 'type': 'svm', 'accuracy': 1e-6},
		{'name': 'KRR', 'type': 'kernelmachine', 'accuracy': 1e-8},
	)

	params={
		'name': 'Gaussian',
		'args': {'key': ('width',), 'val': (1.5,)},
		'feature_class': 'simple',
		'feature_type': 'Real',
		'data': dataop.get_rand()
	}
	output=fileop.get_output(category.KERNEL, params)
	feats=featop.get_simple('Real', params['data'])
	kernel=GaussianKernel(10, *params['args']['val'])

	_loop(regressions, feats, kernel, output)

