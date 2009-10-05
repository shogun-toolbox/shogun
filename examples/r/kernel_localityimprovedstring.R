library("sg")

size_cache <- 10

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# Locality Improved String
print('LocalityImprovedString')

length <- 5
inner_degree <- 5
outer_degree <- inner_degree+2

dump <- sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree)

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
km <- sg('get_kernel_matrix', 'TEST')

