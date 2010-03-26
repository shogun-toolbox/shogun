library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Combined
print('Combined')

dump <- sg('clean_features', 'TRAIN')
dump <- sg('clean_features', 'TEST')
dump <- sg('set_kernel', 'COMBINED', size_cache)
dump <- sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 3, FALSE)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)

km <- sg('get_kernel_matrix', 'TRAIN')

km <- sg('get_kernel_matrix', 'TEST')
