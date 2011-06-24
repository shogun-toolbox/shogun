"""
Common operations on features
"""

import shogun.Features as features
import shogun.Preprocessor as preproc

WORDSTRING_ORDER=3
WORDSTRING_GAP=0
WORDSTRING_REVERSE=False


def get_features(fclass, ftype, data, *args, **kwargs):
	if fclass=='simple':
		return get_simple(ftype, data, *args, **kwargs)
	elif fclass=='string':
		return get_string(ftype, data, *args, **kwargs)
	elif fclass=='string_complex':
		return get_string_complex(ftype, data, *args, **kwargs)
	elif fclass=='wd':
		return get_wd(data, *args, **kwargs)
	else:
		raise ValueError, 'Unknown feature class %s.'%fclass


def get_simple (ftype, data, alphabet=features.DNA, sparse=False):
	"""Return SimpleFeatures.

	@param ftype Feature type, e.g. Real, Byte
	@param data Train/test data for feature creation
	@param alphabet Alphabet for feature creation
	@param sparse Is feature sparse?
	@return Dict with SimpleFeatures train/test
	"""

	if ftype=='Byte' or ftype=='Char':
		train=eval('features.'+ftype+'Features(alphabet)')
		test=eval('features.'+ftype+'Features(alphabet)')
		train.copy_feature_matrix(data['train'])
		test.copy_feature_matrix(data['test'])

	else:
		train=eval('features.'+ftype+"Features(data['train'])")
		test=eval('features.'+ftype+"Features(data['test'])")

	if sparse:
		sparse_train=eval('features.Sparse'+ftype+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('features.Sparse'+ftype+'Features()')
		sparse_test.obtain_from_simple(test)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':train, 'test':test}


def get_string (ftype, data, alphabet=features.DNA):
	"""Return StringFeatures.

	@param ftype Feature type, e.g. Real, Byte
	@param data Train/test data for feature creation
	@param alphabet Alphabet for feature creation
	@return Dict with StringFeatures train/test
	"""

	train=eval('features.String'+ftype+"Features(data['train'], alphabet)")
	test=eval('features.String'+ftype+"Features(data['test'], alphabet)")
	return {'train':train, 'test':test}


def get_string_complex (ftype, data, alphabet=features.DNA,
	order=WORDSTRING_ORDER, gap=WORDSTRING_GAP, reverse=WORDSTRING_REVERSE):
	"""Return complex StringFeatures.

	@param ftype Feature type, e.g. RealFeature, ByteFeature
	@param data Train/test data for feature creation
	@param alphabet Alphabet for feature creation
	@param order Order of the feature
	@param gap Gap of the feature
	@param reverse Is feature reverse?
	@return Dict with complex StringFeatures train/test
	"""

	feats={}

	charfeat=features.StringCharFeatures(data['train'], alphabet)
	feat=eval('features.String'+ftype+'Features(alphabet)')
	feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats['train']=feat

	charfeat=features.StringCharFeatures(data['test'], alphabet)
	feat=eval('features.String'+ftype+'Features(alphabet)')
	feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats['test']=feat

	if ftype=='Word' or ftype=='Ulong':
		name='Sort'+ftype+'String'
		return add_preproc(name, feats)
	else:
		return feats


def get_wd (data, order=WORDSTRING_ORDER):
	"""Return WDFeatures.

	@param data Train/test data for feature creation
	@param order Order of the feature
	@return Dict with WDFeatures train/test
	"""

	feats={}

	charfeat=features.StringCharFeatures(data['train'], features.DNA)
	bytefeat=features.StringByteFeatures(features.RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['train']=features.WDFeatures(bytefeat, order, order)

	charfeat=features.StringCharFeatures(data['test'], features.DNA)
	bytefeat=features.StringByteFeatures(features.RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['test']=features.WDFeatures(bytefeat, order, order)

	return feats


def add_preproc (name, feats, *args):
	"""Add a preprocessor to the given features.

	@param name Name of the preprocessor
	@param feats Features train/test
	@param *args Variable argument list of the preprocessor
	@return Dict with features having a preprocessor applied
	"""

	fun=eval('preproc.'+name)
	preproc=fun(*args)
	preproc.init(feats['train'])
	feats['train'].add_preproc(preproc)
	feats['train'].apply_preproc()
	feats['test'].add_preproc(preproc)
	feats['test'].apply_preproc()

	return feats

