library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
fm_train_dna <- t(as.matrix(read.table('../data/fm_train_dna.dat')))
fm_test_dna <- t(as.matrix(read.table('../data/fm_test_dna.dat')))

# combined
print('Combined')

kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures(fm_train_real)
subkfeats_test <- RealFeatures(fm_test_real)
subkernel <- GaussianKernel(as.integer(10), 1.6)
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

subkfeats_train <- StringCharFeatures("DNA")
dump <- subkfeats_train$set_features(subkfeats_train, fm_train_dna)
subkfeats_test <- StringCharFeatures("DNA")
dump <- subkfeats_test$set_features(subkfeats_test, fm_test_dna)
degree <- as.integer(3)
subkernel <- FixedDegreeStringKernel(as.integer(10), degree)
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

subkfeats_train <- StringCharFeatures("DNA")
dump <- subkfeats_train$set_features(subkfeats_train, fm_train_dna)
subkfeats_test <- StringCharFeatures("DNA")
dump <- subkfeats_test$set_features(subkfeats_test, fm_test_dna)
subkernel <- LocalAlignmentStringKernel(as.integer(10))
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

dump <- kernel$init(kernel, feats_train, feats_train)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
