"""
Utilities for testing
"""

from modshogun import *
from modshogun import *
from modshogun import *
from modshogun import *
from numpy import *


def check_accuracy (accuracy, **kwargs):
	acc=double(accuracy)
	output=[]

	for key, val in kwargs.items():
		if val is not None:
			output.append('%s: %e' % (key, val))
	print(', '.join(output)+' <--- accuracy: %e' % accuracy)

	for val in kwargs.values():
		if val is not None and val>acc:
			return False

	return True


def get_args (indata, prefix=''):
	"""
	Slightly esoteric function to build a tuple to be used as argument to
	constructor calls.

	Python dicts are not ordered, so we have to look at the number in
	the parameter's name and insert items appropriately into an
	ordered list
	"""

	ident=prefix+'arg'
	# need to pregenerate list for using indices in loop
	args=len(indata)*[None]

	for i in indata:
		if i.find(ident)==-1:
			continue

		try:
			idx=int(i[len(ident)])
		except ValueError:
			raise ValueError('Wrong indata data %s: "%s"!' % (ident, i))

		if i.find('_distance')!=-1: # DistanceKernel
			args[idx]=eval(indata[i]+'()')
		else:
			try:
				args[idx]=eval(indata[i])
			except TypeError: # no bool
				args[idx]=indata[i]

	# weed out superfluous Nones
	return [arg for arg in args if arg is not None]


def get_features(indata, prefix=''):
	fclass=prefix+'feature_class'
	if indata[fclass]=='simple':
		return get_feats_simple(indata, prefix)
	elif indata[fclass]=='string':
		return get_feats_string(indata, prefix)
	elif indata[fclass]=='string_complex':
		return get_feats_string_complex(indata, prefix)
	elif indata[fclass]=='wd':
		return get_feats_wd(indata, prefix)
	else:
		raise ValueError('Unknown feature class %s!'%indata[prefix+'feature_class'])


def get_feats_simple (indata, prefix=''):
	ftype=indata[prefix+'feature_type']

	# have to explicitely set data type for numpy if not real
	as_types={
		'Byte': ubyte,
		'Real': double,
		'Word': ushort
	}
	data_train=indata[prefix+'data_train'].astype(as_types[ftype])
	data_test=indata[prefix+'data_test'].astype(as_types[ftype])

	if ftype=='Byte' or ftype=='Char':
		alphabet=eval(indata[prefix+'alphabet'])
		ftrain=eval(ftype+'Features(alphabet)')
		ftest=eval(ftype+'Features(alphabet)')
		ftrain.copy_feature_matrix(data_train)
		ftest.copy_feature_matrix(data_test)
	else:
		ftrain=eval(ftype+'Features(data_train)')
		ftest=eval(ftype+'Features(data_test)')

	if (indata[prefix+'name'].find('Sparse')!=-1 or (
		'classifier_type' in indata and \
			indata['classifier_type']=='linear')):
		sparse_train=eval('Sparse'+ftype+'Features()')
		sparse_train.obtain_from_simple(ftrain)

		sparse_test=eval('Sparse'+ftype+'Features()')
		sparse_test.obtain_from_simple(ftest)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':ftrain, 'test':ftest}


def get_feats_string (indata, prefix=''):
	ftype=indata[prefix+'feature_type']
	alphabet=eval(indata[prefix+'alphabet'])
	feats={
		'train': eval('String'+ftype+'Features(alphabet)'),
		'test': eval('String'+ftype+'Features(alphabet)')
	}
	feats['train'].set_features(list(indata[prefix+'data_train'][0]))
	feats['test'].set_features(list(indata[prefix+'data_test'][0]))

	return feats


def get_feats_string_complex (indata, prefix=''):
	alphabet=eval(indata[prefix+'alphabet'])
	feats={
		'train': StringCharFeatures(alphabet),
		'test': StringCharFeatures(alphabet)
	}

	if alphabet==CUBE: # data_{train,test} ints due to test.py:_read_matrix
		data_train=[str(x) for x in list(indata[prefix+'data_train'][0])]
		data_test=[str(x) for x in list(indata[prefix+'data_test'][0])]
	else:
		data_train=list(indata[prefix+'data_train'][0])
		data_test=list(indata[prefix+'data_test'][0])

	feats['train'].set_features(data_train)
	feats['test'].set_features(data_test)

	feat=eval('String'+indata[prefix+'feature_type']+ \
		"Features(alphabet)")
	feat.obtain_from_char(feats['train'], indata[prefix+'order']-1,
		indata[prefix+'order'], indata[prefix+'gap'],
		eval(indata[prefix+'reverse']))
	feats['train']=feat

	feat=eval('String'+indata[prefix+'feature_type']+ \
		"Features(alphabet)")
	feat.obtain_from_char(feats['test'], indata[prefix+'order']-1,
		indata[prefix+'order'], indata[prefix+'gap'],
		eval(indata[prefix+'reverse']))
	feats['test']=feat

	if indata[prefix+'feature_type']=='Word' or \
		indata[prefix+'feature_type']=='Ulong':
		name='Sort'+indata[prefix+'feature_type']+'String'
		return add_preprocessor(name, feats)
	else:
		return feats


def get_feats_wd (indata, prefix=''):
	order=indata[prefix+'order']
	feats={}

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(list(indata[prefix+'data_train'][0]))
	bytefeat=StringByteFeatures(RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['train']=WDFeatures(bytefeat, order, order)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(list(indata[prefix+'data_test'][0]))
	bytefeat=StringByteFeatures(RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['test']=WDFeatures(bytefeat, order, order)

	return feats


def add_preprocessor(name, feats, *args):
	fun=eval(name)
	preproc=fun(*args)
	preproc.init(feats['train'])
	feats['train'].add_preprocessor(preproc)
	feats['train'].apply_preprocessor()
	feats['test'].add_preprocessor(preproc)
	feats['test'].apply_preprocessor()

	return feats

