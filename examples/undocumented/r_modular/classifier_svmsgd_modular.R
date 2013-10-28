library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# sgd
print('SVMSGD')

realfeat <- RealFeatures()
dummy <- realfeat$set_feature_matrix(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(realfeat)
realfeat <- RealFeatures()
dummy <- realfeat$set_feature_matrix(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(realfeat)

C <- 2.3
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

svm <- SVMSGD(C, feats_train, labels)
#dump <- svm$io$set_loglevel(0)
#dump <- svm$set_epochs(num_iter)
dump <- svm$train()

dump <- svm$set_features(feats_test)
lab <- svm$apply()
out <- lab$get_labels()
