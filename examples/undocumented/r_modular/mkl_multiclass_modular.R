library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_multiclass <- as.real(as.matrix(read.table('../data/label_train_multiclass.dat')))

# MKLMulticlass
print('MKLMulticlass')


kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures()
dump <- subkfeats_train$set_feature_matrix(fm_train_real)
subkfeats_test <- RealFeatures()
dump <- subkfeats_test$set_feature_matrix(fm_test_real)
subkernel <- GaussianKernel(as.integer(10), 1.2)
dump <- feats_train$append_feature_obj(subkfeats_train)
dump <- feats_test$append_feature_obj(subkfeats_test)
dump <- kernel$append_kernel(subkernel)

subkfeats_train <- RealFeatures()
dump <- subkfeats_train$set_feature_matrix(fm_train_real)
subkfeats_test <- RealFeatures()
dump <- subkfeats_test$set_feature_matrix(fm_test_real)
subkernel <- LinearKernel()
dump <- feats_train$append_feature_obj(subkfeats_train)
dump <- feats_test$append_feature_obj(subkfeats_test)
dump <- kernel$append_kernel(subkernel)

subkfeats_train <- RealFeatures()
dump <- subkfeats_train$set_feature_matrix(fm_train_real)
subkfeats_test <- RealFeatures()
dump <- subkfeats_test$set_feature_matrix(fm_test_real)
subkernel <- PolyKernel(as.integer(10), as.integer(2))
dump <- feats_train$append_feature_obj(subkfeats_train)
dump <- feats_test$append_feature_obj(subkfeats_test)
dump <- kernel$append_kernel(subkernel)
dump <- kernel$init(feats_train, feats_train)

C <- 1.2
epsilon <- 1e-5
mkl_eps <- 0.001
mkl_norm <- 1
num_threads <- as.integer(1)
labels <- MulticlassLabels()
labels$set_labels(label_train_multiclass)

svm <- MKLMulticlass(C, kernel, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$set_mkl_epsilon(mkl_eps)
#dump <- svm$set_mkl_norm(1.5)
dump <- svm$train()

dump <- kernel$init(feats_train, feats_test)
lab <- svm$apply()
out <- lab$get_labels()
