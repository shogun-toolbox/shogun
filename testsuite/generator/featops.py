from shogun.Features import *
from shogun.PreProc import *

WORDSTRING_ORDER=3
WORDSTRING_GAP=0
WORDSTRING_REVERSE=False

def get_simple (type, data, alphabet=DNA, sparse=False):
	type=type.capitalize()
	if type=='Byte' or type=='Char':
		train=eval(type+"Features(data['train'], alphabet)")
		test=eval(type+"Features(data['test'], alphabet)")
	else:
		train=eval(type+"Features(data['train'])")
		test=eval(type+"Features(data['test'])")

	if sparse:
		sparse_train=eval('Sparse'+type+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('Sparse'+type+'Features()')
		sparse_test.obtain_from_simple(test)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':train, 'test':test}

def get_string (type, data, alphabet=DNA):
	type=type.capitalize()
	train=eval('String'+type+"Features(alphabet)")
	train.set_string_features(data['train'])
	test=eval('String'+type+"Features(alphabet)")
	test.set_string_features(data['test'])

	return {'train':train, 'test':test}

def get_wordstring (data, alphabet=DNA, order=WORDSTRING_ORDER,
	gap=WORDSTRING_GAP, reverse=WORDSTRING_REVERSE):

	feats={}

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['train'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat, WORDSTRING_ORDER-1,
		WORDSTRING_ORDER, WORDSTRING_GAP, WORDSTRING_REVERSE)
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	feat=StringCharFeatures(alphabet)
	feat.set_string_features(data['test'])
	wordfeat=StringWordFeatures(feat.get_alphabet());
	wordfeat.obtain_from_char(feat, WORDSTRING_ORDER-1,
		WORDSTRING_ORDER, WORDSTRING_GAP, WORDSTRING_REVERSE)
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return feats


