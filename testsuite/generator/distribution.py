"""Generator for Distribution"""

import numpy
import shogun.Distribution as distribution
import shogun.Library as library

from shogun.Library import Math_init_random
from dataop import INIT_RANDOM

import fileop
import featop
import dataop
import config

def _get_outdata (name, params):
	"""Return data to be written into the testcase's file.

	After computations and such, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.
	
	@param name Distribution method's name
	@param params Gathered data
	@return Dict containing testcase data to be written to file
	"""

	ddata=config.DISTRIBUTION[name]
	outdata={
		'name':name,
		'data_train':numpy.matrix(params['data']['train']),
		'data_test':numpy.matrix(params['data']['test']),
		'data_class':ddata[0][0],
		'data_type':ddata[0][1],
		'feature_class':ddata[1][0],
		'feature_type':ddata[1][1],
		'init_random':dataop.INIT_RANDOM,
		'distribution_accuracy':ddata[2],
	}

	if ddata[1][0]=='string' or (ddata[1][0]=='simple' and ddata[1][1]=='Char'):
		outdata['alphabet']='DNA'
		outdata['seqlen']=dataop.LEN_SEQ
	elif ddata[1][0]=='string_complex':
		if params.has_key('order'):
			outdata['order']=params['order']
		else:
			outdata['order']=featop.WORDSTRING_ORDER

		if params.has_key('alphabet'):
			outdata['alphabet']=params['alphabet']
		else:
			outdata['alphabet']='DNA'

		outdata['gap']=featop.WORDSTRING_GAP
		outdata['reverse']=featop.WORDSTRING_REVERSE
		outdata['seqlen']=dataop.LEN_SEQ
		outdata['feature_obtain']=ddata[1][2]

	optional=['N', 'M', 'pseudo', 'num_examples',
		'derivatives', 'likelihood',
		'best_path', 'best_path_state']
	for opt in optional:
		if params.has_key(opt):
			outdata['distribution_'+opt]=params[opt]

	return outdata

def _get_derivatives (dist, num_vec):
	"""Return the sum of all log_derivatives of a distribution.

	@param distribution Distribution to query
	@param num_vec Number of feature vectors
	@return Sum of all log_derivatives
	"""

	num_param=dist.get_num_model_parameters()
	derivatives=0

	for i in xrange(num_param):
		for j in xrange(num_vec):
			val=dist.get_log_derivative(i, j)
			if val!=-numpy.inf and val!=numpy.nan: # only sparse matrix!
				derivatives+=val

	return derivatives

def _run (name):
	"""Run generator for a specific distribution method.

	@param name Name of the distribtuion method
	"""

	# put some constantness into randomness
	Math_init_random(INIT_RANDOM)

	params={
		'data':dataop.get_dna(),
	}
	feats=featop.get_string_complex('Word', params['data'])

	fun=eval('distribution.'+name)
	dist=fun(feats['train'])
	dist.train()

	params['likelihood']=dist.get_log_likelihood_sample()
	params['derivatives']=_get_derivatives(
		dist, feats['train'].get_num_vectors())

	outdata=_get_outdata(name, params)
	fileop.write(config.C_DISTRIBUTION, outdata)

def _run_hmm ():
	"""Run generator for Hidden-Markov-Model."""

	# put some constantness into randomness
	Math_init_random(INIT_RANDOM)

	params={
		'N':3,
		'M':6,
		'num_examples':4,
		'pseudo':1e-10,
		'order':1,
		'alphabet':'CUBE',
	}

	params['data']=dataop.get_cubes(params['num_examples'],1)
	feats=featop.get_string_complex(
		'Word', params['data'], eval('library.'+params['alphabet']),
		params['order'])
	hmm=distribution.HMM(
		feats['train'], params['N'], params['M'], params['pseudo'])
	hmm.train()
	hmm.baum_welch_viterbi_train(distribution.BW_NORMAL)

	params['likelihood']=hmm.get_log_likelihood_sample()
	params['derivatives']=_get_derivatives(
		hmm, feats['train'].get_num_vectors())


	params['best_path']=0
	params['best_path_state']=0
	for i in xrange(params['num_examples']):
		params['best_path']+=hmm.best_path(i)
		for j in xrange(params['N']):
			params['best_path_state']+=hmm.get_best_path_state(i, j)

	outdata=_get_outdata('HMM', params)
	fileop.write(config.C_DISTRIBUTION, outdata)

def run ():
	"""Run generator for all distribution methods."""

	_run('Histogram')
	_run('LinearHMM')
	_run_hmm()
