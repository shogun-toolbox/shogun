from numpy import *
from shogun.Distribution import *
from shogun.Features import RealFeatures

import fileop
import featop
import dataop
from distributionlist import DISTRIBUTIONLIST

def _get_output_params (name, params, data):
	ddata=DISTRIBUTIONLIST[name]
	output={
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
		output['alphabet']='DNA'
		output['seqlen']=dataop.LEN_SEQ

	for k, v in params.iteritems():
		output['distribution_'+k]=v

	return output

def _run_histogram ():
	data=dataop.get_rand(type=ushort)
	feats=featop.get_simple('Word', data)

	histogram=Histogram(feats['train'])
	histogram.train()

	params={}
	output=_get_output_params('Histogram', params, data)
	fileop.write(fileop.T_DISTRIBUTION, output)

def _run_hmm ():
	data=dataop.get_dna()
	feats=featop.get_string_complex('Word', data)
	params={
		'N':1,
		'M':2,
		'pseudo':1.,
	}
	model=Model()

	hmm=HMM(params['N'], params['M'], model, params['pseudo'])
	#hmm.set_observations(feats['train'])
	hmm.train()

	output=_get_output_params('HMM', params, data)
	fileop.write(fileop.T_DISTRIBUTION, output)

def run ():
	_run_histogram()
	_run_hmm()
