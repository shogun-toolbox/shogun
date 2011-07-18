library("sg")

size_cache <- 10
svm_nu <- 0.1
epsilon <- 1e-5
use_bias <- TRUE
width <- 2.1

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# LibSVMOneClass
print('LibSVMOneClass')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('new_classifier', 'LIBSVM_ONECLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('svm_nu', svm_nu)
dump <- sg('svm_use_bias', use_bias)
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
result <- sg('classify')
