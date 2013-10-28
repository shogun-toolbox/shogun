library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# linear
print('Linear')

feats_train <- RealFeatures()
dummy <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dummy <- feats_test$set_feature_matrix(fm_test_real)
scale <- 1.2

kernel <- LinearKernel(feats_train, feats_train)
dump <- kernel$set_normalizer(AvgDiagKernelNormalizer(scale))
km_train <- kernel$get_kernel_matrix()

kernel <- LinearKernel(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
