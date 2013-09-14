library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# distance
print('Distance')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.7
distance <- EuclideanDistance()

kernel <- DistanceKernel(feats_train, feats_test, width, distance)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
