library("sg")

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# Jensen Metric
print('JensenMetric')

dump <- sg('set_distance', 'JENSEN', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dm <- sg('get_distance_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
dm <- sg('get_distance_matrix', 'TEST')
