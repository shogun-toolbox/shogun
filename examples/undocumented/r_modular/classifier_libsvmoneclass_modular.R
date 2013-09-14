library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# libsvm oneclass
print('LibSVMOneClass')

feats_train <- RealFeatures()
dump <- feats_train$set_features(feats_train, fm_train_real)
feats_test <- RealFeatures()
dump <- feats_test$set_features(feats_test, fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 1.017
epsilon <- 1e-5
num_threads <- as.integer(4)

svm <- LibSVMOneClass(C, kernel)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train(svm)

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$apply(svm)
out <- lab$get_labels(lab)

