from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import Alphabet,DNA, Labels
from shogun.Classifier import *
from numpy import array, zeros, int32, arange, double, ones

klist=open('../../testsuite/klist.py', 'r')
KLIST=eval(klist.read())
klist.close()

PREFIX_SVM='svm_'

########################################################################
# kernel computation
########################################################################

def _kernel (feats, input, accuracy, params):
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

def _kernel_svm (input, accuracy, params):
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
		except:
			args.append(input[p])
	return args

def _realkernel (input, accuracy, params):
	feats={'train':RealFeatures(input['data_train']),
		'test':RealFeatures(input['data_test'])}
	return _kernel(feats, input, accuracy, params)

def _stringkernel (input, accuracy, params):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))
	return _kernel(feats, input, accuracy, params)

def _wordkernel (input, accuracy, params):
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

	return _kernel(feats, input, accuracy, params)


########################################################################
# public
########################################################################

def test (input):
	if input['name'].startswith(PREFIX_SVM):
		input['name']=input['name'][len(PREFIX_SVM):]
		kernel=KLIST[input['name']]
		return _kernel_svm(input, kernel[1], kernel[2])
	else:
		kernel=KLIST[input['name']]
		fun=eval('_'+kernel[0]+'kernel')
		return fun(input, kernel[1], kernel[2])
