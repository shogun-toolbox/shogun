library(shogun)

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(read.table('../data/label_train_dna42.dat'))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat'))
label_train_multiclass <- as.real(read.table('../data/label_train_multiclass.dat'))

# lda
print('LDA')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

gamma <- 3
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

lda <- LDA(gamma, feats_train, labels)
dump <- lda$parallel$set_num_threads(lda$parallel, num_threads)
dump <- lda$train()

dump <- lda$get_bias(lda)
dump <- lda$get_w(lda)
dump <- lda$set_features(lda, feats_test)
lab <- lda$classify(lda)
out <- lab$get_labels(lab)
