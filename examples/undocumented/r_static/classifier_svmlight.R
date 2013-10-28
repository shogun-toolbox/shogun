library("sg")

size_cache <- 10
C <- 10
epsilon <- 1e-5
use_bias <- TRUE

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.double(as.matrix(read.table('../data/label_train_dna.dat')))
degree <- 20

# SVM Light
dosvmlight <- function()
{
	print('SVMLight')

	dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	dump <- sg('set_kernel',  'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)

	dump <- sg('set_labels', 'TRAIN', label_train_dna)

	dump <- sg('new_classifier', 'SVMLIGHT')
	dump <- sg('svm_epsilon', epsilon)
	dump <- sg('c', C)
	dump <- sg('svm_use_bias', use_bias)
	dump <- sg('train_classifier')

	dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
	result <- sg('classify')
}
try(dosvmlight())
