"""
Utilities for testing
"""

from numpy import double, ushort, ubyte, matrix
from sg import sg

SIZE_CACHE=10


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


def get_args (indata, prefix=''):
	# python dicts are not ordered, so we have to look at the number in
	# the parameter's name and insert items appropriately into an
	# ordered list

	# need to pregenerate list for using indices in loop
	ident=prefix+'arg'
	args=len(indata)*[None]

	for i in indata:
		if i.find(ident)==-1:
			continue

		try:
			idx=int(i[len(ident)])
		except ValueError:
			raise ValueError, 'Wrong indata data %s: "%s"!' % (ident, i)

		if i.find('_distance')!=-1: # DistanceKernel
			args[idx]=indata[i]
		else:
			try:
				args[idx]=eval(indata[i])
			except TypeError: # no bool
				args[idx]=indata[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)


def set_features (indata, prefix):
	uppername=indata[prefix+'name'].upper()
	if uppername=='COMBINED' or uppername=='CUSTOM':
		return

	if uppername.startswith('SPARSE'):
		raise NotImplementedError, 'Sparse features not supported yet.'
	elif indata.has_key(prefix+'type') and \
		indata[prefix+'type']=='linear':
		raise NotImplementedError, 'Linear classifiers with sparse features not supported yet.'

	indata_train=indata[prefix+'data_train']
	indata_test=indata[prefix+'data_test']

	if indata.has_key(prefix+'alphabet'):
		alphabet=indata[prefix+'alphabet']
		if alphabet=='RAWBYTE':
			raise NotImplementedError, 'Alphabet RAWBYTE not supported yet.'

		if alphabet=='RAWDNA':
			data_train=list(indata_train[0])
			sg('set_features', 'TRAIN', data_train, 'DNA')
			sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'BYTE')
			data_test=list(indata_test[0])
			sg('set_features', 'TEST', data_test, 'DNA')
			sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'BYTE')
		elif alphabet=='CUBE':
			data_train=[str(x) for x in list(indata_train[0])]
			sg('set_features', 'TRAIN', data_train, alphabet)
			data_test=[str(x) for x in list(indata_test[0])]
			sg('set_features', 'TEST', data_test, alphabet)
		else:
			data_train=list(indata_train[0])
			sg('set_features', 'TRAIN', data_train, alphabet)
			data_test=list(indata_test[0])
			sg('set_features', 'TEST', data_test, alphabet)

	elif indata.has_key('data'): # CustomKernel
		sg('set_features', 'TRAIN',
			indata[prefix+'data'].astype(ushort))
		sg('set_features', 'TEST',
			indata[prefix+'data'].astype(ushort))

	else:
		as_types={
			'Word': ushort,
			'Real': double,
			'Byte': ubyte
		}
		as_type=as_types[indata[prefix+'feature_type']]
		sg('set_features', 'TRAIN', indata_train.astype(as_type))
		sg('set_features', 'TEST', indata_test.astype(as_type))

	convert_features_and_add_preproc(indata, prefix)


def set_and_train_distance (indata, do_train=True):
	prefix='distance_'
	dargs=get_args(indata, prefix)
	dname=fix_distance_name_inconsistency(indata[prefix+'name'])
	sg('set_distance', dname, indata[prefix+'feature_type'].upper(), *dargs)

def set_and_train_kernel (indata, do_train=True):
	prefix='kernel_'
	kargs=get_args(indata, prefix)
	kname=fix_kernel_name_inconsistency(indata[prefix+'name'])

	if indata.has_key(prefix+'arg0_size'):
		size=kargs[0]
		kargs=kargs[1:]
	else:
		size=SIZE_CACHE

	if kname=='POLY' and indata.has_key(prefix+'normalizer'):
		kargs.append(True)

	if kname=='DISTANCE':
		dname=fix_distance_name_inconsistency(kargs.pop())
		# FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL')
		sg('set_kernel', kname, size, *kargs)
	else:
		sg('set_kernel', kname, indata[prefix+'feature_type'].upper(), size,
			*kargs)

def convert_features_and_add_preproc (indata, prefix):
	# having order implies having gap and reverse
	if not indata.has_key(prefix+'order'):
		return

	if indata[prefix+'feature_type']=='Ulong':
		type='ULONG'
	elif indata[prefix+'feature_type']=='Word':
		type='WORD'
	else:
		return

	order=indata[prefix+'order']
	sg('add_preproc', 'SORT'+type+'STRING')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, indata[prefix+'gap'], indata[prefix+'reverse'])
	sg('attach_preproc', 'TRAIN')

	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, indata[prefix+'gap'], indata[prefix+'reverse'])
	sg('attach_preproc', 'TEST')


# fix inconsistency in modular/static interfaces
def fix_kernel_name_inconsistency (kname):
	kname=kname.upper()
	if kname=='LOCALITYIMPROVEDSTRING':
		return 'LIK'
	elif kname=='SIMPLELOCALITYIMPROVEDSTRING':
		return 'SLIK'
	elif kname=='WORDMATCH':
		return 'MATCH'
	elif kname=='WEIGHTEDDEGREEPOSITIONSTRING':
		return 'WEIGHTEDDEGREEPOS'
	elif kname=='COMMULONGSTRING':
		return 'COMMSTRING'
	elif kname=='COMMWORDSTRING':
		return 'COMMSTRING'
	elif kname=='WEIGHTEDCOMMWORDSTRING':
		return 'WEIGHTEDCOMMSTRING'
	elif kname.endswith('WORDSTRING'):
		return kname.split('WORDSTRING')[0]
	elif kname.endswith('STRING'):
		return kname.split('STRING')[0]
	elif kname.endswith('WORD'):
		return kname.split('WORD')[0]
	else:
		return kname


def fix_distance_name_inconsistency (dname):
	dname=dname.upper()
	if dname.endswith('WORDDISTANCE'):
		return dname.split('WORDDISTANCE')[0]
	elif dname.endswith('DISTANCE'):
		return dname.split('DISTANCE')[0]
	elif dname.endswith('METRIC'):
		return dname.split('METRIC')[0]
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

def fix_preproc_name_inconsistency (pname):
	return pname.upper()

def fix_regression_name_inconsistency (rname):
	return rname.upper()
