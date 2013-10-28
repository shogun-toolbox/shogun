library(shogun)

# linear byte
print('LinearByte')

feats_train <- ByteFeatures(CSVFile('../data/fm_train_byte.dat'))
feats_test <- ByteFeatures(CSVFile('../data/fm_test_byte.dat'))

kernel <- LinearKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
kernel <- LinearKernel(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
