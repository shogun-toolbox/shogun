"""Generator for Distribution"""

import numpy
import shogun.Distribution as distribution
import shogun.Features as features

from shogun.Library import Math_init_random
from dataop import INIT_RANDOM

import fileop
import featop
import dataop
import category

PREFIX='distribution_'


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
		'name': name,
		'accuracy': 1e-7,
		'data':dataop.get_dna(),
		'alphabet': 'DNA',
		'feature_class': 'string_complex',
		'feature_type': 'Word'
	}
	output=fileop.get_output(category.DISTRIBUTION, params)
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	dfun=eval('distribution.'+name)
	dist=dfun(feats['train'])
	dist.train()

	output[PREFIX+'likelihood']=dist.get_log_likelihood_sample()
	output[PREFIX+'derivatives']=_get_derivatives(
		dist, feats['train'].get_num_vectors())

	fileop.write(category.DISTRIBUTION, output)


def _run_hmm ():
	"""Run generator for Hidden-Markov-Model."""

	# put some constantness into randomness
	Math_init_random(INIT_RANDOM)

	num_examples=4
	params={
		'name': 'HMM',
		'accuracy': 1e-6,
		'N': 3,
		'M': 6,
		'num_examples': num_examples,
		'pseudo': 1e-10,
		'order': 1,
		'alphabet': 'CUBE',
		'feature_class': 'string_complex',
		'feature_type': 'Word',
		'data': dataop.get_cubes(num_examples, 1)
	}
	output=fileop.get_output(category.DISTRIBUTION, params)

	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'],
		eval('features.'+params['alphabet']), params['order'])

	hmm=distribution.HMM(
		feats['train'], params['N'], params['M'], params['pseudo'])
	hmm.train()
	hmm.baum_welch_viterbi_train(distribution.BW_NORMAL)

	output[PREFIX+'likelihood']=hmm.get_log_likelihood_sample()
	output[PREFIX+'derivatives']=_get_derivatives(
		hmm, feats['train'].get_num_vectors())

	output[PREFIX+'best_path']=0
	output[PREFIX+'best_path_state']=0
	for i in xrange(num_examples):
		output[PREFIX+'best_path']+=hmm.best_path(i)
		for j in xrange(params['N']):
			output[PREFIX+'best_path_state']+=hmm.get_best_path_state(i, j)

	fileop.write(category.DISTRIBUTION, output)


def run ():
	"""Run generator for all distribution methods."""

	_run('Histogram')
	_run('LinearHMM')
	_run_hmm()
