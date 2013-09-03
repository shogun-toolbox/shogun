library(modshogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# sparse_linear
print('SparseLinear')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
scale <- 1.1

kernel <- SparseLinearKernel()
dump <- kernel$set_normalizer(kernel, AvgDiagKernelNormalizer(scale))
dump <- kernel$init(kernel, feats_train, feats_train)
