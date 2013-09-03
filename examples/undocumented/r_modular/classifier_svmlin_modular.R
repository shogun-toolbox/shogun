library(modshogun)

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# svm lin
print('SVMLin')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 1.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- SVMLin(C, feats_train, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_bias_enabled(svm, TRUE)
dump <- svm$train(svm)

dump <- svm$set_features(svm, feats_test)
dump <- svm$get_bias(svm)
dump <- svm$get_w(svm)
lab <- svm$apply(svm)
out <- lab$get_labels(lab)
