library(modshogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# sgd
print('SVMSGD')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 2.3
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- SVMSGD(C, feats_train, labels)
#dump <- svm$io$set_loglevel(svm$io, 0)
#dump <- svm$set_epochs(num_iter)
dump <- svm$train(svm)

dump <- svm$set_features(svm, feats_test)
lab <- svm$apply(svm)
out <- lab$get_labels(lab)
