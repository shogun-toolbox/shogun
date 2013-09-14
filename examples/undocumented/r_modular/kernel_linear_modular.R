library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# linear
print('Linear')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
scale <- 1.2

kernel <- LinearKernel()
dump <- kernel$set_normalizer(kernel, AvgDiagKernelNormalizer(scale))
dump <- kernel$init(kernel, feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
