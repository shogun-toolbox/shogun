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


#
# real features
#


# BrayCurtis Distance
print('BrayCurtisDistance')

dump <- sg('set_distance', 'BRAYCURTIS', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Euclidian Distance
print('EuclidianDistance')

dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Canberra Metric
print('CanberraMetric')

dump <- sg('set_distance', 'CANBERRA', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Chebyshew Metric
print('ChebyshewMetric')

dump <- sg('set_distance', 'CHEBYSHEW', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# ChiSquare Distance
print('ChiSquareDistance')

dump <- sg('set_distance', 'CHISQUARE', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Cosine Distance
print('CosineDistance')

dump <- sg('set_distance', 'COSINE', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Geodesic Metric
print('GeodesicMetric')

dump <- sg('set_distance', 'GEODESIC', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Jensen Metric
print('JensenMetric')

dump <- sg('set_distance', 'JENSEN', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Manhattan Metric
print('ManhattanMetric')

dump <- sg('set_distance', 'MANHATTAN', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Minkowski Metric
print('MinkowskiMetric')

k <- 3

dump <- sg('set_distance', 'MINKOWSKI', 'REAL', k)

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Tanimoto Distance
print('TanimotoDistance')

dump <- sg('set_distance', 'TANIMOTO', 'REAL')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')



#
# complex string features
#

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true


# Canberra Word Distance
print('CanberraWordDistance')

dump <- sg('set_distance', 'CANBERRA', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Hamming Word Distance
print('HammingWordDistance')

dump <- sg('set_distance', 'HAMMING', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Manhattan Word Distance
print('ManhattanWordDistance')

dump <- sg('set_distance', 'MANHATTAN', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')

