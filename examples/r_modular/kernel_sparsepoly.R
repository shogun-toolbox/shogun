library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

# sparse_poly
print('SparsePoly')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
size_cache <- as.integer(10)
degree <- as.integer(3)
inhomogene <- TRUE

kernel <- SparsePolyKernel(feats_train, feats_train, size_cache, degree,
	inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
