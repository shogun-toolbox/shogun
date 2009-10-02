library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

# weighted_degree_string
print('WeightedDegreeString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(20)

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

#weights <- arange(1,degree+1,dtype <- double)[::-1]/ \
#	sum(arange(1,degree+1,dtype <- double))
#kernel$set_wd_weights(weights)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
