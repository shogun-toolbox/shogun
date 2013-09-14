library(shogun)

fm_train_byte <- as.matrix(read.table('../data/fm_train_byte'))
fm_test_byte <- as.matrix(read.table('../data/fm_test_byte'))

# linear byte
print('LinearByte')

num_feats <- 11
feats_train <- ByteFeatures(RAWBYTE)
feats_train$set_feature_matrix(traindata_byte)

feats_test <- ByteFeatures(RAWBYTE)
feats_test$set_feature_matrix(testdata_byte)

kernel <- LinearByteKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
