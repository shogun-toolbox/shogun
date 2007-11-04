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
		'test':RealFeatures(input['data_train'])}

	kfun=eval(input['name'])
	k=kfun(feats['train'],feats['train'], *args, **kwargs)
	#numvec = feat.get_num_vectors();
	l=Labels(double(input['labels']))
	svm=SVMLight(input['size_'], k, l) # labels: 0.1, 1, 10
	svm.train()
	alphas=svm.get_alphas()
	train=max(alphas-input['alphas'])

	#gsv = svm.get_support_vectors()
	#max2 = max(testgsv-dict['alphas']) # eigtl. 0/1 index

	#feat = RealFeatures(dict['data_test'])
	#gk.init(feat, feat)
	#out = svm.classify().get_labels() #e-4/5 precision

	#bias = svm.get_bias() #e-4/5 precision

	# checken gegen generierte referenz
	#max2 = max(abs(dict['svm_out']-out))

	print "train: %e, accuracy: %e" % (train, accuracy)

	if train<accuracy:
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
	return _realkernel(input, 1e-6, input['size_'], input['degree'],
		eval(input['inhom']), eval(input['use_norm']))

def weighteddegreestring (input):
	degree=input['degree']
	weights=arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
	return _stringkernel(input, 1e-10, degree, weights=weights)

def weighteddegreepositionstring (input):
	return _stringkernel(input, 1e-8, input['degree'],
		ones(input['seqlen'], dtype=int32))

def commwordstring (input):
	return _wordkernel(input, 1e-9)

def svm_gaussian (input):
	return _kernel_svm(input, 1e-8, input['width_'], input['size_'])
