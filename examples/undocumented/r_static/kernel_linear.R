library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Linear
print('Linear')

dump <- sg('set_kernel', 'LINEAR', 'REAL', size_cache)
dump <- sg('set_features', 'TRAIN', fm_train_real)

dump <- sg('set_kernel_normalization', 'SQRTDIAG')
km1 <- sg('get_kernel_matrix', 'TRAIN')
dump <- sg('set_kernel_normalization', 'AVGDIAG')
km2 <- sg('get_kernel_matrix', 'TRAIN')


#dump <- sg('set_features', 'TEST', fm_test_real)
#km <- sg('get_kernel_matrix', 'TEST')
