"""
Common operations on features
"""

import shogun.Features as features
import shogun.PreProc as preproc
import shogun.Library as library

WORDSTRING_ORDER=3
WORDSTRING_GAP=0
WORDSTRING_REVERSE=False

def get_simple (ftype, data, alphabet=library.DNA, sparse=False):
	"""Return SimpleFeatures.

	@param ftype Feature type, e.g. Real, Byte
	@param data Train/test data for feature creation
	@param alphabet Alphabet for feature creation
	@param sparse Is feature sparse?
	@return Dict with SimpleFeatures train/test
	"""

	if ftype=='Byte' or ftype=='Char':
		train=eval('features.'+ftype+"Features(data['train'], alphabet)")
		test=eval('features.'+ftype+"Features(data['test'], alphabet)")
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

def get_string (ftype, data, alphabet=library.DNA):
	"""Return StringFeatures.

	@param ftype Feature type, e.g. Real, Byte
	@param data Train/test data for feature creation
	@param alphabet Alphabet for feature creation
	@return Dict with StringFeatures train/test
	"""

	train=eval('features.String'+ftype+"Features(alphabet)")
	train.set_string_features(data['train'])
	test=eval('features.String'+ftype+"Features(alphabet)")
	test.set_string_features(data['test'])

	return {'train':train, 'test':test}

def get_string_complex (ftype, data, alphabet=library.DNA,
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

	charfeat=features.StringCharFeatures(alphabet)
	charfeat.set_string_features(data['train'])
	feat=eval('features.String'+ftype+'Features(charfeat.get_alphabet())')
	feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats['train']=feat

	charfeat=features.StringCharFeatures(alphabet)
	charfeat.set_string_features(data['test'])
	feat=eval('features.String'+ftype+'Features(charfeat.get_alphabet())')
	feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats['test']=feat

	if ftype=='Word' or ftype=='Ulong':
		name='Sort'+ftype+'String'
		return add_preproc(name, feats)
	else:
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

