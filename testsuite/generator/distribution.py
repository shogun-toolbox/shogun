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

	optional=['N', 'M', 'pseudo']
	for opt in optional:
		if params.has_key(opt):
			outdata['distribution_'+opt]=params[opt]

	return outdata

def _run_histogram ():
	params={
		'data':dataop.get_rand(dattype=ushort),
	}
	feats=featop.get_simple('Word', params['data'])

	histo=Histogram(feats['train'])
	histo.train()

	outdata=_get_outdata('Histogram', params)
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
	_run_histogram()
	_run_hmm()
