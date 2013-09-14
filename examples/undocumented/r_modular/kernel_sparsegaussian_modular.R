library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# sparse_gaussian
print('SparseGaussian')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
width <- 1.1

kernel <- SparseGaussianKernel(feats_train, feats_train, width)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
