library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna.dat')))

# plugin_estimate
print('PluginEstimate w/ HistogramWord')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)

pie <- PluginEstimate()
labels <- Labels(label_train_dna)
dump <- pie$set_labels(pie, labels)
dump <- pie$set_features(pie, feats_train)
dump <- pie$train(pie)

kernel <- HistogramWordStringKernel(feats_train, feats_train, pie)
km_train <- kernel$get_kernel_matrix()

dump <- kernel$init(kernel, feats_train, feats_test)
dump <- pie$set_features(pie, feats_test)
km_test <- kernel$get_kernel_matrix()
