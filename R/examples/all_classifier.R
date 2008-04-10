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
use_bias <- 1

traindat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
testdat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
trainlab_one <- c(rep(-1,num),rep(1,num))
trainlab_multi <- c(rep(0,num/2), rep(1,num/2), rep(2,num/2), rep(3,num/2))


#
# kernel-based SVMs
#

# SVM Light
print('SVMLight')

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

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('send_command', paste('set_kernel WEIGHTEDDEGREE CHAR', size_cache, degree))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_dna)
dump <- sg('send_command', 'new_svm LIGHT')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('send_command', 'init_kernel TEST')
# segfaults
result <- sg('svm_classify')


# LibSVM
print('LibSVM')

width <- 2.1

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm LIBSVM')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# GPBTSVM
print('GPBTSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm GPBTSVM')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# MPDSVM
print('MPDSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm MPDSVM')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# LibSVM MultiClass
print('LibSVMMultiClass')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_multi)
dump <- sg('send_command', 'new_svm LIBSVM_MULTICLASS')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# LibSVMOneClass
print('LibSVMOneClass')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm LIBSVM_ONECLASS')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# GMNPSVM
print('GMNPSVM')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm GMNPSVM')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


#
# run with batch or linadd on LibSVM
#

# LibSVM batch
print('LibSVM batch')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_svm LIBSVM')
dump <- sg('send_command', paste('svm_epsilon', epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', paste('svm_use_bias', use_bias))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_kernel TEST')

objective <- sg('get_svm_objective')
dump <- sg('send_command', 'use_batch_computation 1')
dump <- sg('send_command', 'use_linadd 1')
result <- sg('svm_classify')


#
# misc classifiers
#

# Perceptron
print('Perceptron')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_classifier PERCEPTRON')
dump <- sg('send_command', 'train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
result <- sg('classify')


# KNN
print('KNN')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'set_distance EUCLIDIAN REAL')
dump <- sg('send_command', 'init_distance TRAIN')
dump <- sg('send_command', 'new_knn')
dump <- sg('send_command', 'train_knn')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'init_distance TEST')
result <- sg('classify')


# LDA
print('LDA')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('set_labels', 'TRAIN', trainlab_one)
dump <- sg('send_command', 'new_classifier LDA')
dump <- sg('send_command', 'train_classifier')

dump <- sg('set_features', 'TEST', testdat_real)
result <- sg('classify')

