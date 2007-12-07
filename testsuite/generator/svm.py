from numpy import *
from numpy.random import rand
from shogun.Kernel import *
from shogun.Features import Labels
from shogun.Classifier import *
from shogun.Library import E_WD

import fileop
import featop
import dataop
from svmlist import SVMLIST

L_TWOCLASS=0
L_SERIES=1

T_KERNEL=0
T_DISTANCE=1
T_LINEAR=2

def _get_output_params(name, params, type, typedata):
	output={
		'data_train':matrix(typedata['data']['train']),
		'data_test':matrix(typedata['data']['test']),
		'svmparam_C':params['C'],
		'svmparam_epsilon':params['epsilon'],
		'svmparam_num_threads':params['num_threads'],
		'svmparam_classified':params['classified'],
		'svmparam_accuracy':SVMLIST[name][0],
		'svmparam_type':SVMLIST[name][1],
	}

	if params['labels'] is not None:
		output['svmparam_labels']=params['labels']

	if type==T_LINEAR:
		output['feature_class']='simple'
		output['feature_type']='Real'
		output['data_type']='double'
		if params.has_key('bias'):
			output['svmparam_bias']=params['bias']
	else:
		output['svmparam_tube_epsilon']=params['tube_epsilon']
		output['svmparam_bias']=params['bias']
		output['svmparam_alphas']=params['alphas']
		output['svmparam_support_vectors']=params['support_vectors']

		if type==T_KERNEL:
			output['kname']=typedata['kname']
			output.update(
				fileop.get_output_params(typedata['kname'], typedata['kargs']))

	return output

def _compute (name, params, type, typedata):
	svmfun=eval(name)

	lab=None
	if typedata['labels'] is not None:
		# essential to wrap in array(), will segfault sometimes otherwise
		lab=Labels(array(typedata['labels']))
		params['labels']=typedata['labels']

	if type==T_KERNEL:
		typedata['k'].parallel.set_num_threads(params['num_threads'])
		typedata['k'].init(typedata['feats']['train'], typedata['feats']['train'])
		if lab is None:
			svm=svmfun(params['C'], typedata['k'])
		else:
			svm=svmfun(params['C'], typedata['k'], lab)

	elif type==T_LINEAR:
		svm=svmfun(params['C'], typedata['feats']['train'], lab)
	else:
		return None

	svm.parallel.set_num_threads(params['num_threads'])
	svm.set_epsilon(params['epsilon'])

	if type!=T_LINEAR:
		svm.set_tube_epsilon(params['tube_epsilon'])
	else:
		svm.set_bias_enabled(typedata['bias_enabled'])

	svm.train()

	if typedata.has_key('bias_enabled') and typedata['bias_enabled']:
		params['bias']=svm.get_bias()
	elif type!=T_LINEAR:
		params['bias']=svm.get_bias()
		params['alphas']=svm.get_alphas()
		params['support_vectors']=svm.get_support_vectors()

		if type==T_KERNEL:
			typedata['k'].init(typedata['feats']['train'], typedata['feats']['test'])

	params['classified']=svm.classify().get_labels()

	return [name, _get_output_params(name, params, type, typedata)]

def _run (svms, type, typedata):
	if type==T_KERNEL:
		kfun=eval(typedata['kname']+'Kernel')
		# FIXME: cache size has to go....
		typedata['k']=kfun(10, *typedata['kargs'])
		# FIXME: NASTY NASTY NASTY! but WeightedStringKernel is a bit inconsistent
		# in constructors, so have to get rid of first arg EWDKernType
		if typedata['kname']=='WeightedDegreeString':
			typedata['kargs']=typedata['kargs'][1:]

	num_vec=typedata['feats']['train'].get_num_vectors();
	if typedata['ltype']==L_TWOCLASS:
		typedata['labels']=rand(num_vec).round()*2-1
	elif typedata['ltype']==L_SERIES:
		typedata['labels']=[double(x) for x in xrange(num_vec)]
	else:
		typedata['labels']=None

	for name in svms:
		params={'C':.017, 'epsilon':1e-5, 'tube_epsilon':1e-2, 'num_threads':1}
		fileop.write(_compute(name, params, type, typedata))
		params['C']=.23
		fileop.write(_compute(name, params, type, typedata))
		params['C']=1.5
		fileop.write(_compute(name, params, type, typedata))
		params['C']=30
		fileop.write(_compute(name, params, type, typedata))
		params['epsilon']=1e-4
		fileop.write(_compute(name, params, type, typedata))
		params['tube_epsilon']=1e-3
		fileop.write(_compute(name, params, type, typedata))
		params['num_threads']=16
		fileop.write(_compute(name, params, type, typedata))

def _run_svm_kernel ():
	svms=['SVMLight', 'LibSVM', 'GPBTSVM', 'MPDSVM']
	typedata={
		'kname':'Gaussian',
		'kargs':[1.5],
		'data':dataop.get_rand(),
		'ltype':L_TWOCLASS
	}
	typedata['feats']=featop.get_simple('Real', typedata['data'])
	_run(svms, T_KERNEL, typedata)

	svms=['LibSVMMultiClass', 'GMNPSVM']
	typedata['ltype']=L_SERIES
	_run(svms, T_KERNEL, typedata)

#	svms=['LibSVMOneClass']
#	typedata['ltype']=None
#	_run(svms, T_KERNEL, typedata)

	svms=['SVMLight', 'GPBTSVM']
	typedata['kname']='Linear'
	typedata['ltype']=L_TWOCLASS
	_run(svms, T_KERNEL, typedata)

	typedata['data']=dataop.get_dna()
	typedata['feats']=featop.get_string('Char', typedata['data'])
	typedata['kname']='WeightedDegreeString'
	typedata['kargs']=[E_WD, 3, 0]
	_run(svms, T_KERNEL, typedata)
	typedata['kname']='WeightedDegreePositionString'
	typedata['kargs']=[20]
	_run(svms, T_KERNEL, typedata)

	typedata['kargs']=[False, FULL_NORMALIZATION]
	typedata['kname']='CommWordString'
	typedata['feats']=featop.get_string_complex('Word', typedata['data'])
	_run(svms, T_KERNEL, typedata)
	typedata['kname']='CommUlongString'
	typedata['feats']=featop.get_string_complex('Ulong', typedata['data'])
	_run(svms, T_KERNEL, typedata)

def _run_svm_linear ():
	#svms=['SubGradientSVM', 'SVMOcas']
	svms=['SVMOcas']
	typedata={
		'data':dataop.get_rand(),
		'ltype':L_TWOCLASS,
		'bias_enabled':False,
	}
	typedata['feats']=featop.get_simple('Real', typedata['data'], sparse=True)
	_run(svms, T_LINEAR, typedata)

	svms=['LibLinear', 'SVMLin']
	typedata['bias_enabled']=True
	_run(svms, T_LINEAR, typedata)

def run ():
	fileop.TYPE='SVM'

	_run_svm_kernel()
	_run_svm_linear()



