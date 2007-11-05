from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import Alphabet,DNA, Labels
from shogun.Classifier import *
from numpy import array, zeros, int32, arange, double, ones

def _kernel (name, feats, kms, accuracy, *args, **kwargs):
	kfun=eval(name)
	k=kfun(feats['train'], feats['train'], *args, **kwargs)
	train=max(abs(kms['train']-k.get_kernel_matrix()).flat)
	k.init(feats['train'], feats['test'])
	test=max(abs(kms['test']-k.get_kernel_matrix()).flat)
	print "train: %e, test: %e, accuracy: %e" % (train, test, accuracy)

	if train<accuracy and test<accuracy:
		return True

	return False

def _realkernel (input, accuracy, *args, **kwargs):
	feats={'train':RealFeatures(input['data_train']),
		'test':RealFeatures(input['data_test'])}
	kms={'train':input['km_train'], 'test':input['km_test']}
	return _kernel(input['name'], feats, kms, accuracy, *args, **kwargs)

def _stringkernel (input, accuracy, *args, **kwargs):
	feats={'train':StringCharFeatures(eval(input['alphabet'])),
		'test':StringCharFeatures(eval(input['alphabet']))}
	feats['train'].set_string_features(list(input['data_train'][0]))
	feats['test'].set_string_features(list(input['data_test'][0]))
	kms={'train':input['km_train'], 'test':input['km_test']}
	return _kernel(input['name'], feats, kms, accuracy, *args, **kwargs)

def _wordkernel (input, accuracy, *args, **kwargs):
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

	kms={'train':input['km_train'], 'test':input['km_test']}
	return _kernel(input['name'], feats, kms, accuracy, *args, **kwargs)

def _kernel_svm (input, accuracy, *args, **kwargs):
	feats={'train':RealFeatures(input['data_train']),
		'test':RealFeatures(input['data_test'])}

	kfun=eval(input['name'])
	k=kfun(feats['train'],feats['train'], *args, **kwargs)
	train=max(abs(input['km_train']-k.get_kernel_matrix()).flat)

	l=Labels(double(input['labels']))
	svm=SVMLight(input['size_'], k, l) # labels: 0.1, 1, 10
	svm.train()
	check_alphas=max(abs(svm.get_alphas()-input['alphas']))
	check_bias=abs(svm.get_bias()-input['bias'])
	check_sv=max(abs(svm.get_support_vectors()-input['support_vectors']))

	k.init(feats['train'], feats['test'])
	test=max(abs(input['km_test']-k.get_kernel_matrix()).flat)
	check_classified=max(abs(svm.classify().get_labels()-input['classified']))

	print "check_alphas: %e, check_bias: %e, check_sv: %e, check_classified: %e, train: %e, test: %e, accuracy: %e" % (check_alphas, check_bias, check_sv, check_classified, train, test, accuracy)

	if (check_alphas<accuracy and
		check_bias<accuracy and
		check_sv<accuracy and
		check_classified<accuracy and
		train<accuracy and
		test<accuracy):
		return True

	return False

def gaussian (input):
	return _realkernel(input, 1e-8, input['width_'], input['size_'])

def linear (input):
	return _realkernel(input, 1e-8, input['scale'])

def chi2 (input):
	return _realkernel(input, 1e-8, input['size_'])

def sigmoid (input):
	return _realkernel(input, 1e-9, input['size_'], input['gamma_'],
		input['coef0'])

def poly (input):
	return _realkernel(input, 1e-6, input['degree'],
		eval(input['inhomogene']), eval(input['use_normalization']),
		input['size_'])

def weighteddegreestring (input):
	return _stringkernel(input, 1e-10, input['degree'],
		weights=input['weights'])

def weighteddegreepositionstring (input):
	shifts=ones(input['seqlen'], dtype=int32)
	return _stringkernel(input, 1e-8, input['degree'], shifts)

def commwordstring (input):
	return _wordkernel(input, 1e-9, eval(input['use_sign']),
		input['normalization'], input['size_'])

def svm_gaussian (input):
	return _kernel_svm(input, 1e-8, input['width_'], input['size_'])
