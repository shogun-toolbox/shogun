"""
Generator for Distribution
"""

from numpy import *
from shogun.Distribution import *

import fileop
import featop
import dataop
from config import DISTRIBUTION, C_DISTRIBUTION

def _get_outdata (name, params):
	ddata=DISTRIBUTION[name]
	outdata={
		'name':name,
		'data_train':matrix(params['data']['train']),
		'data_test':matrix(params['data']['test']),
		'data_class':ddata[0][0],
		'data_type':ddata[0][1],
		'feature_class':ddata[1][0],
		'feature_type':ddata[1][1],
		'distribution_accuracy':ddata[2],
	}

	if ddata[1][0]=='string' or (ddata[1][0]=='simple' and ddata[1][1]=='Char'):
		outdata['alphabet']='DNA'
		outdata['seqlen']=dataop.LEN_SEQ
	elif ddata[1][0]=='string_complex':
		outdata['order']=featop.WORDSTRING_ORDER
		outdata['gap']=featop.WORDSTRING_GAP
		outdata['reverse']=featop.WORDSTRING_REVERSE
		outdata['alphabet']='DNA'
		outdata['seqlen']=dataop.LEN_SEQ
		outdata['feature_obtain']=ddata[1][2]

	optional=['N', 'M', 'pseudo',
		'derivatives', 'likelihood']
	for opt in optional:
		if params.has_key(opt):
			outdata['distribution_'+opt]=params[opt]

	return outdata

def _run (name):
	params={
		'data':dataop.get_dna(),
	}
	feats=featop.get_string_complex('Word', params['data'])

	fun=eval(name)
	distribution=fun(feats['train'])
	distribution.train()

	num_examples=feats['train'].get_num_vectors()
	num_param=distribution.get_num_model_parameters()
	params['derivatives']=0
	params['likelihood']=0

	for i in xrange(num_examples):
		for j in xrange(num_param):
			params['derivatives']+=distribution.get_log_derivative(j, i)
		params['likelihood']+=distribution.get_log_likelihood_example(i)

	outdata=_get_outdata(name, params)
	fileop.write(C_DISTRIBUTION, outdata)

def _run_hmm ():
	params={
		'N':1,
		'M':2,
		'pseudo':1.,
		'data':dataop.get_dna(),
	}
	feats=featop.get_string_complex('Word', params['data'])
	model=Model()

	hmm=HMM(params['N'], params['M'], model, params['pseudo'])
	#hmm.set_observations(feats['train'])
	hmm.train()

	outdata=_get_outdata('HMM', params)
	fileop.write(C_DISTRIBUTION, outdata)

def run ():
	_run('Histogram')
	_run('LinearHMM')
	_run_hmm()
