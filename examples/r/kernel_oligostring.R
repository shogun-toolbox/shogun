library("sg")

size_cache <- 10

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# Oligo String
print('OligoString')

k <- 3
width <- 1.2

dump <- sg('set_kernel', 'OLIGO', 'CHAR', size_cache, k, width)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')
