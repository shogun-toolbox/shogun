"""
Generator for Distribution
"""

from numpy import *
from shogun.Distribution import *

import fileop
import featop
import dataop
from config import DISTRIBUTION, C_DISTRIBUTION

def _get_outdata_params (name, params, data):
	ddata=DISTRIBUTION[name]
	outdata={
		'name':name,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test']),
		'data_class':ddata[0][0],
		'data_type':ddata[0][1],
		'feature_class':ddata[1][0],
		'feature_type':ddata[1][1],
		'accuracy':ddata[2],
	}

	if ddata[1][0]=='string' or (ddata[1][0]=='simple' and ddata[1][1]=='Char'):
		outdata['alphabet']='DNA'
		outdata['seqlen']=dataop.LEN_SEQ

	for key, val in params.iteritems():
		outdata['distribution_'+key]=val

	return outdata

def _run_histogram ():
	data=dataop.get_rand(dattype=ushort)
	feats=featop.get_simple('Word', data)

	histo=Histogram(feats['train'])
	histo.train()

	params={}
	outdata=_get_outdata_params('Histogram', params, data)
	fileop.write(C_DISTRIBUTION, outdata)

def _run_hmm ():
	data=dataop.get_dna()
	#feats=featop.get_string_complex('Word', data)
	params={
		'N':1,
		'M':2,
		'pseudo':1.,
	}
	model=Model()

	hmm=HMM(params['N'], params['M'], model, params['pseudo'])
	#hmm.set_observations(feats['train'])
	hmm.train()

	outdata=_get_outdata_params('HMM', params, data)
	fileop.write(C_DISTRIBUTION, outdata)

def run ():
	_run_histogram()
	_run_hmm()
