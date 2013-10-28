library("sg")

size_cache <- 10
order <- 3
gap <- 0
reverse <- 'n'

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.double(as.matrix(read.table('../data/label_train_dna.dat')))

# PluginEstimate
print('PluginEstimate w/ HistogramWord')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

pseudo_pos <- 1e-1
pseudo_neg <- 1e-1

dump <- sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
dump <- sg('set_labels', 'TRAIN', label_train_dna)
dump <- sg('train_estimator')

dump <- sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
km <- sg('get_kernel_matrix', 'TRAIN')

# not supported yet
#	lab=sg('plugin_estimate_classify')
km <- sg('get_kernel_matrix', 'TEST')
