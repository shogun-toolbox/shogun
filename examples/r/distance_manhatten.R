# Explicit examples on how to use the different distances
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# Manhattan Metric
print('ManhattanMetric')

dump <- sg('set_distance', 'MANHATTAN', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dm <- sg('get_distance_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
dm <- sg('get_distance_matrix', 'TEST')

