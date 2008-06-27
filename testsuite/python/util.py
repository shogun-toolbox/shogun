"""
Utilities for testing
"""

from numpy import double, ushort
from sg import sg

SIZE_CACHE=10


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
			args[idx]=indata[i]
		else:
			try:
				args[idx]=eval(indata[i])
			except TypeError: # no bool
				args[idx]=indata[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)


def set_features (indata):
	if indata['name'].startswith('Sparse'):
		raise NotImplementedError, 'Sparse features not supported yet.'
	elif indata.has_key('classifier_type') and \
		indata['classifier_type']=='linear':
		raise NotImplementedError, 'Linear classifiers with sparse features not supported yet.'

	if indata.has_key('alphabet'):
		if indata['alphabet']=='RAWBYTE':
			raise NotImplementedError, 'Alphabet RAWBYTE not supported yet.'

		if indata['alphabet']=='CUBE':
			data_train=[str(x) for x in list(indata['data_train'][0])]
			data_test=[str(x) for x in list(indata['data_test'][0])]
		else:
			data_train=list(indata['data_train'][0])
			data_test=list(indata['data_test'][0])

		sg('set_features', 'TRAIN', data_train, indata['alphabet'])
		sg('set_features', 'TEST', data_test, indata['alphabet'])

	elif indata.has_key('data'):
		sg('set_features', 'TRAIN',
			indata['data'].astype(eval(indata['data_type'])))
		sg('set_features', 'TEST',
			indata['data'].astype(eval(indata['data_type'])))

	elif indata['name'].upper()=='COMBINED':
		pass

	else:
		sg('set_features', 'TRAIN',
			indata['data_train'].astype(eval(indata['data_type'])))
		sg('set_features', 'TEST',
			indata['data_test'].astype(eval(indata['data_type'])))

	convert_features_and_add_preproc(indata)


def set_and_train_distance (indata, do_train=True):
	dargs=get_args(indata, 'distance_arg')

	if indata.has_key('distance_name'):
		dname=fix_distance_name_inconsistency(indata['distance_name'])
	else:
		dname=fix_distance_name_inconsistency(indata['name'])

	sg('set_distance', dname, indata['feature_type'].upper(), *dargs)

	if do_train:
		sg('init_distance', 'TRAIN')


def set_and_train_kernel (indata, do_train=True):
	kargs=get_args(indata, 'kernel_arg')

	if indata.has_key('kernel_name'):
		kname=fix_kernel_name_inconsistency(indata['kernel_name'])
	elif indata.has_key('name_kernel'): # FIXME!!!!
		kname=fix_kernel_name_inconsistency(indata['name_kernel'])
	else:
		kname=fix_kernel_name_inconsistency(indata['name'])

	if kname.find('COMMSTRING')!=-1:
		kargs[1]=fix_normalization_inconsistency(kargs[1])

	if indata.has_key('kernel_arg0_size'):
		size=kargs[0]
		kargs=kargs[1:]
	else:
		size=SIZE_CACHE

	if kname=='DISTANCE':
		dname=fix_distance_name_inconsistency(kargs.pop())
		# FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL')
		sg('set_kernel', kname, size, *kargs)
	else:
		sg('set_kernel', kname, indata['feature_type'].upper(), size, *kargs)

	if do_train:
		sg('init_kernel', 'TRAIN')



def convert_features_and_add_preproc (indata):
	if indata['feature_type']=='Ulong':
		type='ULONG'
	elif indata['feature_type']=='Word':
		type='WORD'
	else:
		return

	if not indata.has_key('order'):
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
	elif kname.endswith('STRING'):
		return kname.split('STRING')[0]
	elif kname.endswith('WORD'):
		return kname.split('WORD')[0]
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
