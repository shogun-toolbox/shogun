# Explicit examples on how to use the different distances
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30

#
# real features
#

traindat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
testdat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
trainlab_real <- c(rep(-1,num),rep(1,num))

# Euclidian Distance
print('EuclidianDistance')

dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Canberra Metric
print('CanberraMetric')

dump <- sg('set_distance', 'CANBERRA', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Chebyshew Metric
print('ChebyshewMetric')

dump <- sg('set_distance', 'CHEBYSHEW', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Geodesic Metric
print('GeodesicMetric')

dump <- sg('set_distance', 'GEODESIC', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Jensen Metric
print('JensenMetric')

dump <- sg('set_distance', 'JENSEN', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Manhattan Metric
print('ManhattanMetric')

dump <- sg('set_distance', 'MANHATTAN', 'REAL')

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Minkowski Metric
print('MinkowskiMetric')

k <- 3

dump <- sg('set_distance', 'MINKOWSKI', 'REAL', k)

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


#
# complex string features
#

getDNA <- function(len, num) {
	acgt <- c('A', 'C', 'G', 'T')
	data <- c()
	for (j in 1:num) {
		str <- '';
		for (i in 1:len) {
			str <- paste(str, sample(acgt, 1), sep='')
		}
		data <- append(data, str)
	}
	data
}

trainlab_dna <- c(rep(-1,num/2),rep(1, num/2))
traindat_dna <- getDNA(len, num)
testdat_dna <- getDNA(len, num+7)

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true


# Canberra Word Distance
print('CanberraWordDistance')

dump <- sg('set_distance', 'CANBERRA', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Hamming Word Distance
print('HammingWordDistance')

dump <- sg('set_distance', 'HAMMING', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')


# Manhattan Word Distance
print('ManhattanWordDistance')

dump <- sg('set_distance', 'MANHATTAN', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dump <- sg('init_distance', 'TRAIN')
dm <- sg('get_distance_matrix')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dump <- sg('init_distance', 'TEST')
dm <- sg('get_distance_matrix')

