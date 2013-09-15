library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# liblinear
print('LibLinear')

realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_train_real)

feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(realfeat)
realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_test_real)

feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(realfeat)

C <- 1.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

svm <- LibLinear(C, feats_train, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$set_bias_enabled(TRUE)
dump <- svm$train()

dump <- svm$set_features(feats_test)
lab <- svm$apply()
out <- lab$get_labels()
