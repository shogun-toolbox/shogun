library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

width <- 1.4

# LogPlusOne
print('LogPlusOne')

dump <- sg('add_preproc', 'LOGPLUSONE')
dump <- sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('attach_preproc', 'TRAIN')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('attach_preproc', 'TEST')
km <- sg('get_kernel_matrix', 'TEST')
