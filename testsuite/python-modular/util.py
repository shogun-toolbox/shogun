"""
Utilities for testing
"""

from shogun.Features import *
from shogun.PreProc import *
from shogun.Distance import *
from shogun.Kernel import *
from numpy import *


def check_accuracy (accuracy, **kwargs):
	acc=double(accuracy)
	output=[]

	for key, val in kwargs.iteritems():
		if val is not None:
			output.append('%s: %e' % (key, val))
	print ', '.join(output)+' <--- accuracy: %e' % accuracy

	for val in kwargs.itervalues():
		if val>acc:
			return False

	return True

def get_args (indata, ident):
	# python dicts are not ordered, so we have to look at the number in
	# the parameter's name and insert items appropriately into an
	# ordered list

	# need to pregenerate list for using indices in loop
	args=len(indata)*[None]

	for i in indata:
		if i.find(ident)==-1:
			continue

		try:
			idx=int(i[len(ident)])
		except ValueError:
			raise ValueError, 'Wrong indata data %s: "%s"!' % (ident, i)

		if i.find('_distance')!=-1: # DistanceKernel
			args[idx]=eval(indata[i]+'()')
		else:
			try:
				args[idx]=eval(indata[i])
			except TypeError: # no bool
				args[idx]=indata[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)


# the kernel lists here are highly dependent on how the generator sets
# them up, be careful!
def get_normalizer (kname, do_normalize):
	name=['Poly']
	if kname in name and not do_normalize:
		return IdentityKernelNormalizer()

	name=['Linear', 'SparseLinear', 'LinearString', 'LinearWord', 'LinearByte']
	if kname in name:
		return AvgDiagKernelNormalizer(-1)

	return False


def get_feats_simple (indata):
	# have to explicitely set data type for numpy if not real
	data_train=indata['data_train'].astype(eval(indata['data_type']))
	data_test=indata['data_test'].astype(eval(indata['data_type']))

	if indata['feature_type']=='Byte' or indata['feature_type']=='Char':
		alphabet=eval(indata['alphabet'])
		ftrain=eval(indata['feature_type']+"Features(alphabet)")
		ftest=eval(indata['feature_type']+"Features(alphabet)")
		ftrain.copy_feature_matrix(data_train)
		ftest.copy_feature_matrix(data_test)
	else:
		ftrain=eval(indata['feature_type']+"Features(data_train)")
		ftest=eval(indata['feature_type']+"Features(data_test)")

	if (indata['name'].find('Sparse')!=-1 or (
		indata.has_key('classifier_type') and indata['classifier_type']=='linear')):
		sparse_train=eval('Sparse'+indata['feature_type']+'Features()')
		sparse_train.obtain_from_simple(ftrain)

		sparse_test=eval('Sparse'+indata['feature_type']+'Features()')
		sparse_test.obtain_from_simple(ftest)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':ftrain, 'test':ftest}

def get_feats_string (indata):
	feats={'train':StringCharFeatures(eval(indata['alphabet'])),
		'test':StringCharFeatures(eval(indata['alphabet']))}
	feats['train'].set_string_features(list(indata['data_train'][0]))
	feats['test'].set_string_features(list(indata['data_test'][0]))

	return feats

def get_feats_string_complex (indata):
	alphabet=eval(indata['alphabet'])
	feats={'train':StringCharFeatures(alphabet),
		'test':StringCharFeatures(alphabet)}

	if alphabet==CUBE: # data_{train,test} ints due to test.py:_read_matrix
		data_train=[str(x) for x in list(indata['data_train'][0])]
		data_test=[str(x) for x in list(indata['data_test'][0])]
	else:
		data_train=list(indata['data_train'][0])
		data_test=list(indata['data_test'][0])

	feats['train'].set_string_features(data_train)
	feats['test'].set_string_features(data_test)

	feat=eval('String'+indata['feature_type']+ \
		"Features(feats['train'].get_alphabet())")
	feat.obtain_from_char(feats['train'], indata['order']-1, indata['order'],
		indata['gap'], eval(indata['reverse']))
	feats['train']=feat

	feat=eval('String'+indata['feature_type']+ \
		"Features(feats['train'].get_alphabet())")
	feat.obtain_from_char(feats['test'], indata['order']-1, indata['order'],
		indata['gap'], eval(indata['reverse']))
	feats['test']=feat

	if indata['feature_type']=='Word' or indata['feature_type']=='Ulong':
		name='Sort'+indata['feature_type']+'String'
		return add_preproc(name, feats)
	else:
		return feats

def add_preproc (name, feats, *args):
	fun=eval(name)
	preproc=fun(*args)
	preproc.init(feats['train'])
	feats['train'].add_preproc(preproc)
	feats['train'].apply_preproc()
	feats['test'].add_preproc(preproc)
	feats['test'].apply_preproc()

	return feats

