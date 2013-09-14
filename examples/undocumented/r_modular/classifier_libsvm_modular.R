library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# libsvm
print('LibSVM')

feats_train <- RealFeatures()
dummy <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dummy <- feats_test$set_feature_matrix(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 1.017
epsilon <- 1e-5
num_threads <- as.integer(2)
labels <- BinaryLabels()
print(label_train_twoclass)
dump <- labels$set_labels(label_train_twoclass)

svm <- LibSVM(C, kernel, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$train()

dump <- kernel$init(feats_train, feats_test)
lab <- svm$apply()
out <- lab$get_labels()
