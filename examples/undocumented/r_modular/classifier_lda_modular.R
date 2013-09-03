library(modshogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# lda
print('LDA')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

gamma <- 3
labels <- Labels(label_train_twoclass)

lda <- LDA(gamma, feats_train, labels)
dump <- lda$train(lda)

dump <- lda$get_bias(lda)
dump <- lda$get_w(lda)
dump <- lda$set_features(lda, feats_test)
lab <- lda$apply(lda)
out <- lab$get_labels(lab)
