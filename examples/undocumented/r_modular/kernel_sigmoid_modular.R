library(modshogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# sigmoid
print('Sigmoid')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
size_cache <- as.integer(10)
gamma <- 1.2
coef0 <- 1.3

kernel <- SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
