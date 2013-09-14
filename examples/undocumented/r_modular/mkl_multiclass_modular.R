library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_multiclass <- as.real(as.matrix(read.table('../data/label_train_multiclass.dat')))

# MKLMulticlass
print('MKLMulticlass')


kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures(fm_train_real)
subkfeats_test <- RealFeatures(fm_test_real)
subkernel <- GaussianKernel(as.integer(10), 1.2)
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures(fm_train_real)
subkfeats_test <- RealFeatures(fm_test_real)
subkernel <- LinearKernel()
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures(fm_train_real)
subkfeats_test <- RealFeatures(fm_test_real)
subkernel <- PolyKernel(as.integer(10), as.integer(2))
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)
dump <- kernel$init(kernel, feats_train, feats_train)

C <- 1.2
epsilon <- 1e-5
mkl_eps <- 0.001 
mkl_norm <- 1
num_threads <- as.integer(1)
labels <- Labels(label_train_multiclass)

svm <- MKLMulticlass(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_mkl_epsilon(svm,mkl_eps)
#dump <- svm$set_mkl_norm(1.5)
dump <- svm$train(svm)

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$apply(svm)
out <- lab$get_labels(lab)
