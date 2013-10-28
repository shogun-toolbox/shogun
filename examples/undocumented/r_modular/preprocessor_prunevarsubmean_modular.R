library(shogun)

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))

#PruneVarSubMean
print('PruneVarSubMean')

feats_train <- RealFeatures()
dump <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dump <- feats_test$set_feature_matrix(fm_test_real)

preproc <- PruneVarSubMean()
dump <- preproc$init(feats_train)
dump <- feats_train$add_preproc(preproc)
dump <- feats_train$apply_preproc()
dump <- feats_test$add_preproc(preproc)
dump <- feats_test$apply_preproc()

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
