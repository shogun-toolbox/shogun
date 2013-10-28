library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# comm_ulong_string
print('CommUlongString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(fm_train_dna)
feats_train <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(charfeat, start, order, gap, reverse)
preproc <- SortUlongString()
dump <- preproc$init(feats_train)
dump <- feats_train$add_preproc(preproc)
dump <- feats_train$apply_preproc()


charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(fm_test_dna)
feats_test <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(preproc)
dump <- feats_test$apply_preproc()

use_sign <- FALSE

kernel <- CommUlongStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
