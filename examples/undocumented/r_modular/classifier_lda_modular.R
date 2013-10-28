library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# lda
print('LDA')

feats_train <- RealFeatures()
dump <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dump <- feats_test$set_feature_matrix(fm_test_real)

gamma <- 3
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

lda <- LDA(gamma, feats_train, labels)
dump <- lda$train()

dump <- lda$get_bias()
dump <- lda$get_w()
dump <- lda$set_features(feats_test)
lab <- lda$apply()
out <- lab$get_labels()
