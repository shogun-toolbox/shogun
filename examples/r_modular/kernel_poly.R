library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

# poly
print('Poly')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
degree <- as.integer(4)
inhomogene <- FALSE

kernel <- PolyKernel(
	feats_train, feats_train, degree, inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
