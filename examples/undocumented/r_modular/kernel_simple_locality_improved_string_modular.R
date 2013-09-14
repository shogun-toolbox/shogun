library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# simple_locality_improved_string
print('SimpleLocalityImprovedString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_features(fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_features(fm_test_dna)
l <- as.integer(5)
inner_degree <- as.integer(5)
outer_degree <- as.integer(7)

kernel <- SimpleLocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
