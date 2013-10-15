library("sg")

size_cache <- 10
C <- 1.2
epsilon <- 1e-5
mkl_eps <- 0.01
mkl_norm <- 1.5

width <- 1.2

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_multiclass <- as.real(as.matrix(read.table('../data/label_train_multiclass.dat')))

# MKL_MULTICLASS
print('MKL_MULTICLASS')

dump <- sg('clean_features', 'TRAIN')
dump <- sg('clean_features', 'TEST')
dump <- sg('set_kernel', 'COMBINED', size_cache)
dump <- sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 2)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)

dump <- sg('set_labels', 'TRAIN', label_train_multiclass)
dump <- sg('new_classifier', 'MKL_MULTICLASS')
dump <- sg('svm_epsilon', epsilon)
dump <- sg('c', C)
dump <- sg('mkl_parameters', mkl_eps, 0, mkl_norm);
dump <- sg('train_classifier')

result <- sg('classify')
