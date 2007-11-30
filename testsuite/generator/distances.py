from numpy import *
from shogun.Distance import *

import fileops
import dataops
import featops
from dlist import DLIST


def _get_output (name, output, args=[]):
	ddata=DLIST[name]

	# for all distances
	output['data_class']=ddata[0][0]
	output['data_type']=ddata[0][1]
	output['feature_class']=ddata[1][0]
	output['feature_type']=ddata[1][1]
	output['accuracy']=ddata[3]

	# specialise a bit
	if ddata[1][0]=='string_complex':
		output['order']=featops.WORDSTRING_ORDER
		output['gap']=featops.WORDSTRING_GAP
		output['reverse']=featops.WORDSTRING_REVERSE
		output['alphabet']='DNA'
		output['seqlen']=dataops.LEN_SEQ
		output['feature_obtain']=ddata[1][2]

	# distance arguments, if any
	for i in range(0, len(args)):
		try:
			pname='dparam'+str(i)+'_'+ddata[2][i]
		except IndexError:
			break

		# a bit awkward to have this specialised cond here:
		if pname.find('distance')!=-1:
			output[pname]=args[i].__class__.__name__
		else:
			output[pname]=args[i]

	return output

def _compute (name, feats, data, *args):
	dfun=eval(name)
	d=dfun(feats['train'], feats['train'], *args)
	dm_train=d.get_distance_matrix()
	d.init(feats['train'], feats['test'])
	dm_test=d.get_distance_matrix()

	output=_get_output(name, {
		'dm_train':dm_train,
		'dm_test':dm_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}, args)

	return [name, output]

def _run_feats_real ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)

	fileops.write(_compute('EuclidianDistance', feats, data))

	feats=featops.get_simple('Real', data, sparse=True)
	#fileops.write(_compute('SparseEuclidianDistance', feats, data))

def _run_feats_string_complex ():
	data=dataops.get_dna()
	feats=featops.get_string_complex('Word', data)

	fileops.write(_compute('CanberraWordDistance', feats, data))
	fileops.write(_compute('HammingWordDistance', feats, data, False))
	fileops.write(_compute('HammingWordDistance', feats, data, True))
	fileops.write(_compute('ManhattanWordDistance', feats, data))

def run ():
	fileops.TYPE='Distance'

	_run_feats_real()
	_run_feats_string_complex()

