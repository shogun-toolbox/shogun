"""
Test Regression
"""

from numpy import double
from modshogun import RegressionLabels
from modshogun import *
from modshogun import *

import util

def _evaluate (indata):
	prefix='kernel_'
	feats=util.get_features(indata, prefix)
	kargs=util.get_args(indata, prefix)
	fun=eval(indata[prefix+'name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], *kargs)

	prefix='regression_'
	kernel.parallel.set_num_threads(indata[prefix+'num_threads'])

	try:
		name = indata[prefix+'name']
		if (name=='KERNELRIDGEREGRESSION'):
			name = 'KernelRidgeRegression'
		
		rfun=eval(name)
	except NameError as e:
		print("%s is disabled/unavailable!"%indata[prefix+'name'])
		return False

	labels=RegressionLabels(double(indata[prefix+'labels']))
	if indata[prefix+'type']=='svm':
		regression=rfun(
			indata[prefix+'C'], indata[prefix+'epsilon'], kernel, labels)
	elif indata[prefix+'type']=='kernelmachine':
		regression=rfun(indata[prefix+'tau'], kernel, labels)
	else:
		return False

	regression.parallel.set_num_threads(indata[prefix+'num_threads'])
	if prefix+'tube_epsilon' in indata:
		regression.set_tube_epsilon(indata[prefix+'tube_epsilon'])

	regression.train()

	alphas=0
	bias=0
	sv=0
	if prefix+'bias' in indata:
		bias=abs(regression.get_bias()-indata[prefix+'bias'])
	if prefix+'alphas' in indata:
		for item in regression.get_alphas().tolist():
			alphas+=item
		alphas=abs(alphas-indata[prefix+'alphas'])
	if prefix+'support_vectors' in indata:
		for item in inregression.get_support_vectors().tolist():
			sv+=item
		sv=abs(sv-indata[prefix+'support_vectors'])

	kernel.init(feats['train'], feats['test'])
	classified=max(abs(
		regression.apply().get_labels()-indata[prefix+'classified']))

	return util.check_accuracy(indata[prefix+'accuracy'], alphas=alphas,
		bias=bias, support_vectors=sv, classified=classified)

########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

