# Explicit examples on how to use the different classifiers
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

size_cache <- 10
C <- 10
epsilon <- 1e-5
use_bias <- TRUE

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
label_train_twoclass <- as.real(as.matrix(read.table('../data/label_train_twoclass.dat')))
label_train_multiclass <- as.real(as.matrix(read.table('../data/label_train_multiclass.dat')))

#
# kernel-based SVMs
#

degree <- 20

# SVM Light
dosvmlight <- function()
{
	print('SVMLight')

	dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	dump <- sg('set_kernel',  'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	dump <- sg('init_kernel', 'TRAIN')

	dump <- sg('set_labels', 'TRAIN', label_train_dna)

	dump <- sg('new_classifier', 'SVMLIGHT')
	dump <- sg('svm_epsilon', epsilon)
	dump <- sg('c', C)
	dump <- sg('svm_use_bias', use_bias)
	dump <- sg('train_classifier')

	dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
	dump <- sg('init_kernel', 'TEST')
	result <- sg('classify')
}
try(dosvmlight())


# LibSVM
print('LibSVM')

width <- 2.1

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LIBSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# GPBTSVM
print('GPBTSVM')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'GPBTSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# MPDSVM
print('MPDSVM')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'MPDSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# LibSVM MultiClass
print('LibSVMMultiClass')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_multiclass)
dump <- sg('new_classifier', 'LIBSVM_MULTICLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# LibSVMOneClass
print('LibSVMOneClass')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LIBSVM_ONECLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# GMNPSVM
print('GMNPSVM')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'GMNPSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


#
# run with batch or linadd on LibSVM
#

# LibSVM batch
print('LibSVM batch')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LIBSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_kernel', 'TEST')

objective <- sg('get_svm_objective')
dump <- sg('use_batch_computation', TRUE)
dump <- sg('use_linadd', TRUE)
result <- sg('classify')


#
# misc classifiers
#

# Perceptron
print('Perceptron')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'PERCEPTRON')
# often does not converge
#dump <- sg('train_classifier')

#dump <- sg('set_features', 'TEST', fm_test_real)
#result <- sg('classify')


# KNN
print('KNN')
k <- 3

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('init_distance', 'TRAIN')
dump <- sg('new_classifier', 'KNN')
dump <- sg('train_classifier', k)

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
result <- sg('classify')


# LDA
print('LDA')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LDA')
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
result <- sg('classify')

