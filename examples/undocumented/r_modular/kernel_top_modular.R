library(shogun)

size_cache=as.integer(0)
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

# top_fisher
print('TOP/Fisher on PolyKernel')

N <- as.integer(3)
M <- as.integer(6)
pseudo <- 1e-1
order <- as.integer(1)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("CUBE")
dump <- charfeat$set_features(fm_train_cube)
wordfeats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- wordfeats_train$obtain_from_char(charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(wordfeats_train)
dump <- wordfeats_train$add_preproc(preproc)
dump <- wordfeats_train$apply_preproc()

charfeat <- StringCharFeatures("CUBE")
dump <- charfeat$set_features(fm_test_cube)
wordfeats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- wordfeats_test$obtain_from_char(charfeat, start, order, gap, reverse)
dump <- wordfeats_test$add_preproc(preproc)
dump <- wordfeats_test$apply_preproc()

pos <- HMM(wordfeats_train, N, M, pseudo)
dump <- pos$train()
dump <- pos$baum_welch_viterbi_train("BW_NORMAL")
neg <- HMM(wordfeats_train, N, M, pseudo)
dump <- neg$train()
dump <- neg$baum_welch_viterbi_train("BW_NORMAL")
pos_clone <- HMM(pos)
neg_clone <- HMM(neg)
dump <- pos_clone$set_observations(wordfeats_test)
dump <- neg_clone$set_observations(wordfeats_test)

feats_train <- TOPFeatures(size_cache, pos, neg, FALSE, FALSE)
feats_test <- TOPFeatures(size_cache, pos_clone, neg_clone, FALSE, FALSE)
kernel <- PolyKernel(feats_train, feats_train, as.integer(1), FALSE)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

feats_train <- FKFeatures(size_cache, pos, neg)
dump <- feats_train$set_opt_a(-1); #estimate prior
feats_test <- FKFeatures(size_cache, pos_clone, neg_clone)
dump <- feats_test$set_a(feats_train$get_a()); #use prior from training data
kernel <- PolyKernel(feats_train, feats_train, as.integer(1), FALSE)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

