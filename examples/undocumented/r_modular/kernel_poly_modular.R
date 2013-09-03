library(modshogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

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
