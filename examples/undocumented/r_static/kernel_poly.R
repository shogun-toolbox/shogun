library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Poly
print('Poly')

degree <- 4
inhomogene <- FALSE
use_normalization <- TRUE

dump <- sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')
