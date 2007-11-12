from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Distance import *
from shogun.Classifier import *
from numpy import *

klist=open('../../testsuite/klist.py', 'r')
KLIST=eval(klist.read())
klist.close()

PREFIX_SVM='svm_'

# numpy is picky about int data types
ASTYPE={
	'Real':double,
	'Word':ushort,
	'Byte':ubyte,
}

########################################################################
# kernel computation
########################################################################

def _kernel (accuracy, params, input, feats):
	kfun=eval(input['name']+'Kernel')
	args=_get_args(input, params)

	k=kfun(feats['train'], feats['train'], *args)
	train=max(abs(input['km_train']-k.get_kernel_matrix()).flat)
	k.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-k.get_kernel_matrix()).flat)
	print "train: %e, test: %e, accuracy: %e" % (train, test, accuracy)

	if train<accuracy and test<accuracy:
		return True

	return False

def _kernel_svm (accuracy, params, input):
	feats={'train':RealFeatures(input['data_train']),
		'test':RealFeatures(input['data_test'])}
	args=_get_args(input, params)

	kfun=eval(input['name']+'Kernel')
	k=kfun(feats['train'], feats['train'], *args)
	l=Labels(double(input['labels']))
	svm=SVMLight(input['C'], k, l)
	svm.train()
	check_alphas=max(abs(svm.get_alphas()-input['alphas']))
	check_bias=abs(svm.get_bias()-input['bias'])
	check_sv=max(abs(svm.get_support_vectors()-input['support_vectors']))

	k.init(feats['train'], feats['test'])
	check_classified=max(abs(svm.classify().get_labels()-input['classified']))

	print "check_alphas: %e, check_bias: %e, check_sv: %e, check_classified: %e, accuracy: %e" % (check_alphas, check_bias, check_sv, check_classified, accuracy)

	if (check_alphas<accuracy and
		check_bias<accuracy and
		check_sv<accuracy and
		check_classified<accuracy):
		return True

	return False

def _get_args (input, params):
	args=[]
	for p in params:
		try:
			args.append(eval(input[p]))
		except TypeError: # no bool
			args.append(input[p])
		except KeyError: # does not exist in input
			pass
	return args

def _is_simple (type):
	if type=='String' or type=='Wordstring':
		return False
	return True

def _feats_simple (type, accuracy, params, input):
	input['data_train']=input['data_train'].astype(ASTYPE[type])
	input['data_test']=input['data_test'].astype(ASTYPE[type])

	if type=='Byte' or type=='Char':
		alphabet=eval(input['alphabet'])
		train=eval(type+"Features(input['data_train'], alphabet)")
		test=eval(type+"Features(input['data_test'], alphabet)")
	else:
		train=eval(type+"Features(input['data_train'])")
		test=eval(type+"Features(input['data_test'])")

	if input['name'].find('Sparse')!=-1:
		sparse_train=eval('Sparse'+type+'Features()')
		sparse_train.obtain_from_simple(train)

		sparse_test=eval('Sparse'+type+'Features()')
		sparse_test.obtain_from_simple(test)

		feats={'train':sparse_train, 'test':sparse_test}
	else:
		feats={'train':train, 'test':test}

	return _kernel(accuracy, params, input, feats)

def _feats_string (accuracy, params, input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))
	return _kernel(accuracy, params, input, feats)

def _feats_wordstring (accuracy, params, input):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))

	wordfeat=StringWordFeatures(feats['train'].get_alphabet());
	wordfeat.obtain_from_char(feats['train'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	preproc = SortWordString();
	preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['train']=wordfeat

	wordfeat=StringWordFeatures(feats['test'].get_alphabet());
	wordfeat.obtain_from_char(feats['test'], input['order']-1, input['order'],
		input['gap'], eval(input['reverse']))
	#preproc = SortWordString();
	#preproc.init(wordfeat);
	wordfeat.add_preproc(preproc)
	wordfeat.apply_preproc()
	feats['test']=wordfeat

	return _kernel(accuracy, params, input, feats)


########################################################################
# public
########################################################################

def test (input):
	if input['name'].startswith(PREFIX_SVM):
		input['name']=input['name'][len(PREFIX_SVM):]
		kernel=KLIST[input['name']]
		return _kernel_svm(kernel[1], kernel[2], input)
	else:
		kernel=KLIST[input['name']]

		if _is_simple(kernel[0]):
			return _feats_simple(kernel[0], kernel[1], kernel[2], input)
		else:
			fun=eval('_feats_'+kernel[0].lower())
			return fun(kernel[1], kernel[2], input)

