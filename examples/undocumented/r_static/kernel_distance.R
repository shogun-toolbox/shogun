library("sg")

size_cache <- 10

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Distance
print('Distance')

width=1.7
dump <- sg('set_distance', 'EUCLIDEAN', 'REAL')
dump <- sg('set_kernel', 'DISTANCE', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km=sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km=sg('get_kernel_matrix', 'TEST')
