# Explicit examples on how to use the different kernels
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

size_cache <- 10

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna.dat')))


#
# real features
#

# CHI2
print('Chi2')

width <- 1.4
dump <- sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


# Const
print('Const')

c <- 23.

dump <- sg('set_kernel', 'CONST', 'REAL', size_cache, c)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


# Diag
print('Diag')

diag=23.
dump <- sg('set_kernel', 'DIAG', 'REAL', size_cache, diag)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


# Gaussian
print('Gaussian')

width <- 1.9

dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


# GaussianShift
print('GaussianShift')

width <- 1.8
max_shift <- 2
shift_step <- 1

dump <- sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


# Linear
print('Linear')

scale <- 1.2
dump <- sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


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


# Sigmoid
print('Sigmoid')

gamma <- 1.2
coef0 <- 1.3

dump <- sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km <- sg('get_kernel_matrix', 'TEST')


#
# string features
#

# Fixed Degree String
print('FixedDegreeString')

degree <- 3

dump <- sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Linear String
print('LinearString')

dump <- sg('set_kernel', 'LINEAR', 'CHAR', size_cache)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Local Alignment String
print('LocalAlignmentString')

dump <- sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')

# Oligo String
print('OligoString')

k <- 3
width <- 1.2

dump <- sg('set_kernel', 'OLIGO', 'CHAR', size_cache, k, width)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')

# Poly Match String
print('PolyMatchString')

degree <- 3
inhomogene <- FALSE

dump <- sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Weighted Degree String
print('WeightedDegreeString')

degree <- 20

dump <- sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Weighted Degree Position String
print('WeightedDegreePositionString')

degree <- 20

dump <- sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Locality Improved String
print('LocalityImprovedString')

length <- 5
inner_degree <- 5
outer_degree <- inner_degree+2

dump <- sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


# Simple Locality Improved String
print('SimpleLocalityImprovedString')

length <- 5
inner_degree <- 5
outer_degree <- inner_degree+2

dump <- sg('set_kernel', 'SLIK', 'CHAR', size_cache, length, inner_degree, outer_degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')


#
# complex string features
#

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true
use_sign <- FALSE
normalization <- 'FULL'


# Comm Word String
print('CommWordString')

dump <- sg('add_preproc', 'SORTWORDSTRING')
dump <- sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
km <- sg('get_kernel_matrix', 'TEST')


# Weighted Comm Word String
print('WeightedCommWordString')

dump <- sg('add_preproc', 'SORTWORDSTRING')
dump <- sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
km <- sg('get_kernel_matrix', 'TEST')


# Comm Ulong String
print('CommUlongString')

dump <- sg('add_preproc', 'SORTULONGSTRING')
dump <- sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
km <- sg('get_kernel_matrix', 'TEST')


#
# misc kernels
#

# Distance
print('Distance')

width=1.7
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('set_kernel', 'DISTANCE', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km=sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km=sg('get_kernel_matrix', 'TEST')


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


# PluginEstimate
print('PluginEstimate w/ HistogramWord')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

pseudo_pos <- 1e-1
pseudo_neg <- 1e-1

dump <- sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
dump <- sg('set_labels', 'TRAIN', label_train_dna)
dump <- sg('train_estimator')

dump <- sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
km <- sg('get_kernel_matrix', 'TRAIN')

# not supported yet
#	lab=sg('plugin_estimate_classify')
km <- sg('get_kernel_matrix', 'TEST')
