# Explicit examples on how to use the different classifiers
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30
size_cache <- 10
C <- 10
epsilon <- 1e-5
use_bias <- TRUE

traindat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
testdat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
trainlab_one <- c(rep(-1,num),rep(1,num))
trainlab_multi <- c(rep(0,num/2), rep(1,num/2), rep(2,num/2), rep(3,num/2))


#
# kernel-based SVMs
#

getDNA <- function(len, num) {
	acgt <- c('A', 'C', 'G', 'T')
	data <- c()
	for (j in 1:num) {
		str <- '';
		for (i in 1:len) {
			str <- paste(str, sample(acgt, 1), sep='')
		}
		data <- append(data, str)
	}
	data
}

trainlab_dna <- c(rep(-1,num/2),rep(1, num/2))
traindat_dna <- getDNA(len, num)
testdat_dna <- getDNA(len, num+7)

degree <- 20

# SVM Light
dosvmlight <- function()
{
	print('SVMLight')

	dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
	dump <- sg('set_kernel',  'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	dump <- sg('init_kernel', 'TRAIN')

	dump <- sg('set_labels', 'TRAIN', trainlab_dna)

	dump <- sg('new_svm', 'LIGHT')
	dump <- sg('svm_epsilon', epsilon)
	dump <- sg('c', C)
	dump <- sg('svm_use_bias', use_bias)
	dump <- sg('train_classifier')

	dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
	dump <- sg('init_kernel', 'TEST')
	result <- sg('classify')
}
try(dosvmlight())


# LibSVM
print('LibSVM')

width <- 2.1

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'LIBSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# GPBTSVM
print('GPBTSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'GPBTSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# MPDSVM
print('MPDSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'MPDSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# LibSVM MultiClass
print('LibSVMMultiClass')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_multi)
dump <- sg('new_svm', 'LIBSVM_MULTICLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# LibSVMOneClass
print('LibSVMOneClass')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'LIBSVM_ONECLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# GMNPSVM
print('GMNPSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'GMNPSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


#
# run with batch or linadd on LibSVM
#

# LibSVM batch
print('LibSVM batch')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_svm', 'LIBSVM')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
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

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_classifier', 'PERCEPTRON')
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
result <- sg('classify')


# KNN
print('KNN')
k <- 3

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('init_distance', 'TRAIN')
dump <- sg('new_classifier', 'KNN')
dump <- sg('train_classifier', k)

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
result <- sg('classify')


# LDA
print('LDA')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('new_classifier', 'LDA')
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
result <- sg('classify')

