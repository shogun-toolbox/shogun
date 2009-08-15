#!/usr/bin/env python
"""
Explicit examples on how to use the different classifiers
"""

from numpy import double, array, floor, concatenate, sign, ones, zeros, char, int
from numpy.random import rand, seed, permutation
from sg import sg

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
label_train_dna=lm.load_labels('../data/label_train_dna.dat')
label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
label_train_multiclass=lm.load_labels('../data/label_train_multiclass.dat')

###########################################################################
# kernel-based SVMs
###########################################################################

def svm_light ():
	print 'SVMLight'

	size_cache=10
	degree=20
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_dna)

	try:
		sg('new_classifier', 'SVMLIGHT')
	except RuntimeError:
		return

	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('init_kernel', 'TEST')
	result=sg('classify')

def libsvm ():
	print 'LibSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'LIBSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def gpbtsvm ():
	print 'GPBTSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'GPBTSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def mpdsvm ():
	print 'MPDSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'MPDSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def libsvm_multiclass ():
	print 'LibSVMMultiClass'

	size_cache=10
	width=2.1
	C=10.
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_multiclass)
	sg('new_classifier', 'LIBSVM_MULTICLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def libsvm_oneclass ():
	print 'LibSVMOneClass'

	size_cache=10
	width=2.1
	C=10.
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('new_classifier', 'LIBSVM_ONECLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

def gmnpsvm ():
	print 'GMNPSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'GMNPSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')
	result=sg('classify')

###########################################################################
# run with batch or linadd on LibSVM
###########################################################################

def do_batch_linadd ():
	print 'LibSVM batch'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'LIBSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_kernel', 'TEST')

	objective=sg('get_svm_objective')
	sg('use_batch_computation', True)
	sg('use_linadd', True)
	result=sg('classify')

###########################################################################
# misc classifiers
###########################################################################

def perceptron ():
	print 'Perceptron'

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'PERCEPTRON')
	# often does not converge, mind your data!
	#sg('train_classifier')

	#sg('set_features', 'TEST', fm_test_real)
	#result=sg('classify')

def knn ():
	print 'KNN'

	k=3

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_classifier', 'KNN')
	sg('train_classifier', k)

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	result=sg('classify')

def lda ():
	print 'LDA'

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'LDA')
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	svm_light()
	libsvm()
	gpbtsvm()
	mpdsvm()
	libsvm_multiclass()
	libsvm_oneclass()
	gmnpsvm()

	do_batch_linadd()

	perceptron()
	knn()
	lda()
