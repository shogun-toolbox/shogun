from shogun.Features import RealFeatures, CharFeatures, StringCharFeatures, StringWordFeatures
from shogun.Kernel import *
from shogun.PreProc import *
from shogun.Features import Alphabet,DNA, Labels
from shogun.Classifier import *
from numpy import array, zeros, int32, arange, double, ones

def _kernel (name, feats, kms, accuracy, *args, **kwargs):
	kfun=eval(name)

	# FIXME temporary until interface to C is fixed
	if name.find('WeightedDegree') == -1:
		k=kfun(*args, **kwargs)
		k.init(feats['train'], feats['train'])
	else:
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
	k=kfun(*args, **kwargs)
	k.init(feats['train'], feats['train'])
	train=max(abs(input['km_train']-k.get_kernel_matrix()).flat)

	l=Labels(double(input['labels']))
	svm=SVMLight(input['size_'], k, l)
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
	return _realkernel(input, 1e-8, input['size_'], input['width_'])

def linear (input):
	return _realkernel(input, 1e-8, input['size_'], input['scale'])

def chi2 (input):
	return _realkernel(input, 1e-8, input['size_'])

def sigmoid (input):
	return _realkernel(input, 1e-9, input['size_'], input['gamma_'],
		input['coef0'])

def poly (input):
	return _realkernel(input, 1e-6, input['size_'], input['degree'],
		eval(input['inhomogene']), eval(input['use_normalization']))

def weighteddegreestring (input):
	# FIXME temporary until interface to C is fixed
#	return _stringkernel(input, 1e-10, input['size_'], input['weights'],
#	input['degree'], input['max_mismatch'], input['use_normalization'],
#	input['block_computation'], input['mkl_stepsize'],
#	input['which_degree'])
	return _stringkernel(input, 1e-10, input['degree'],
		input['max_mismatch'], eval(input['use_normalization']),
		eval(input['block_computation']), input['mkl_stepsize'],
		input['which_degree'], input['weights'], input['size_'])

def weighteddegreepositionstring (input):
	# FIXME temporary until interface to C is fixed
#	return _stringkernel(input, 1e-8, input['size_'], input['weights'],
#		input['degree'], input['max_mismatch'], input['shift'],
#		len(input['shift']), input['use_normalization'],
#		input['mkl_stepsize'])
	return _stringkernel(input, 1e-8, input['degree'], input['shift'],
		eval(input['use_normalization']), input['max_mismatch'],
		input['mkl_stepsize'], input['size_'])

def localityimprovedstring (input):
	return _stringkernel(input, 1e-8, input['size_'], input['length'],
		input['inner_degree'], input['outer_degree'])

def fixeddegreestring (input):
	return _stringkernel(input, 1e-10, input['size_'], input['degree'])

def linearstring (input):
	return _stringkernel(input, 1e-8, input['size_'],
		eval(input['do_rescale']) , input['scale'])

def commwordstring (input):
	return _wordkernel(input, 1e-9, input['size_'],
		eval(input['use_sign']), input['normalization'])

def weightedcommwordstring (input):
	return _wordkernel(input, 1e-9, input['size_'],
		eval(input['use_sign']), input['normalization'])

def svm_gaussian (input):
	return _kernel_svm(input, 1e-8, input['size_'], input['width_'])
