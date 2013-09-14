library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# gaussian_shift
print('GaussianShift')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.8
max_shift <- as.integer(2)
shift_step <- as.integer(1)

kernel <- GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
