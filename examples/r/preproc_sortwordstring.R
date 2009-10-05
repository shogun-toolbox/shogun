library("sg")

size_cache <- 10

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

order <- 3
gap <- 0
reverse <- 'n'
use_sign <- FALSE
normalization <- 'FULL'

# Comm Word String
print('CommWordString')

dump <- sg('add_preproc', 'SORTWORDSTRING')
dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')

dump <- sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization)
km <- sg('get_kernel_matrix', 'TRAIN')
km <- sg('get_kernel_matrix', 'TEST')
