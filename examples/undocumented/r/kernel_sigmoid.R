library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Sigmoid
print('Sigmoid')

gamma <- 1.2
coef0 <- 1.3

dump <- sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')
