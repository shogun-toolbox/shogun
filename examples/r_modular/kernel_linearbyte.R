library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

# linear byte
print('LinearByte')

num_feats <- 11
feats_train <- ByteFeatures(RAWBYTE)
feats_train$copy_feature_matrix(traindata_byte)

feats_test <- ByteFeatures(RAWBYTE)
feats_test$copy_feature_matrix(testdata_byte)

kernel <- LinearByteKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
