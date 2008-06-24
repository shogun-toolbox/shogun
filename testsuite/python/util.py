"""
Utilities for testing
"""

from numpy import double
from sg import sg

CACHE_SIZE=10


def check_accuracy (accuracy, **kwargs):
	acc=double(accuracy)
	output=[]

	for key, val in kwargs.iteritems():
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


def set_features (indata):
	if indata.has_key('alphabet'):
		sg('set_features', 'TRAIN',
			list(indata['data_train'][0]), indata['alphabet'])
		sg('set_features', 'TEST',
			list(indata['data_test'][0]), indata['alphabet'])
	else:
		sg('set_features', 'TRAIN',
			indata['data_train'].astype(eval(indata['data_type'])))
		sg('set_features', 'TEST',
			indata['data_test'].astype(eval(indata['data_type'])))


def set_distance (indata):
	dargs=get_args(indata, 'distance_arg')
	dname=fix_distance_name_inconsistency(indata['distance_name'])
	sg('set_distance', dname, indata['feature_type'].upper(),
		CACHE_SIZE, *dargs)
	sg('init_distance', 'TRAIN')


def set_kernel (indata):
	kargs=get_args(indata, 'kernel_arg')
	kname=fix_kernel_name_inconsistency(indata['kernel_name'])

	if kname=='COMMSTRING':
		kargs[1]=fix_normalization_inconsistency(kargs[1])
		convert_features_and_add_preproc(indata)

	sg('set_kernel', kname, indata['feature_type'].upper(),
		CACHE_SIZE, *kargs)
	sg('init_kernel', 'TRAIN')


def convert_features_and_add_preproc (indata):
	if indata['feature_type']=='Ulong':
		type='ULONG'
	elif indata['feature_type']=='Word':
		type='WORD'
	else:
		return

	order=indata['order']
	sg('add_preproc', 'SORT'+type+'STRING')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, indata['gap'], indata['reverse'])
	sg('attach_preproc', 'TRAIN')

	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, indata['gap'], indata['reverse'])
	sg('attach_preproc', 'TEST')


# fix inconsistency in modular/static interfaces
def fix_kernel_name_inconsistency (kname):
	kname=kname.upper()
	if kname=='WEIGHTEDDEGREESTRING':
		return 'WEIGHTEDDEGREE'
	elif kname=='WEIGHTEDDEGREEPOSITIONSTRING':
		return 'WEIGHTEDDEGREEPOS'
	elif kname=='COMMULONGSTRING':
		return 'COMMSTRING'
	elif kname=='COMMWORDSTRING':
		return 'COMMSTRING'
	else:
		return kname

def fix_normalization_inconsistency (normalization):
	if normalization==1:
		return 'SQRT'
	elif normalization==2:
		return 'FULL'
	elif normalization==3:
		return 'SQRTLEN'
	elif normalization==4:
		return 'LEN'
	elif normalization==5:
		return 'SQLEN'
	else:
		return 'NO'

def fix_distance_name_inconsistency (dname):
	dname=dname.upper()
	if dname.endswith('DISTANCE'):
		return dname.split('DISTANCE')[0]
	else:
		return dname


def fix_classifier_name_inconsistency (cname):
	cname=cname.upper()
	if cname.startswith('LIBSVM') and len(cname)>len('LIBSVM'):
		return 'LIBSVM_'+cname.split('LIBSVM')[1]
	else:
		return cname


def fix_clustering_name_inconsistency (cname):
	return cname.upper()
