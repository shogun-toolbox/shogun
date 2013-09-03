library(modshogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# weighted_comm_word_string
print('WeightedCommWordString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- TRUE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

kernel <- WeightedCommWordStringKernel(feats_train, feats_train, use_sign)
km_train <- kernel$get_kernel_matrix()

kernel <- WeightedCommWordStringKernel(feats_train, feats_test, use_sign)
km_test <- kernel$get_kernel_matrix()
