library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# sparse_linear
print('SparseLinear')

feat <- RealFeatures()
dummy <- feat$set_feature_matrix(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feat)
feat <- RealFeatures()
dummy <- feat$set_feature_matrix(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feat)
scale <- 1.1

kernel <- LinearKernel()
dump <- kernel$set_normalizer(AvgDiagKernelNormalizer(scale))
dump <- kernel$init(feats_train, feats_train)
