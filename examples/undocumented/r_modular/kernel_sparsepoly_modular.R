library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# sparse_poly
print('SparsePoly')

feat <- RealFeatures()
dummy <- feat$set_feature_matrix(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feat)
feat <- RealFeatures()
dummy <- feat$set_feature_matrix(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feat)
size_cache <- as.integer(10)
degree <- as.integer(3)
inhomogene <- TRUE

kernel <- PolyKernel(feats_train, feats_train, size_cache, degree,
	inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
