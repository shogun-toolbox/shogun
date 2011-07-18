library("sg")

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

order <- 3
gap <- 0
reverse <- 'n'

# Hamming Word Distance
print('HammingWordDistance')

dump <- sg('set_distance', 'HAMMING', 'WORD')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')
dm <- sg('get_distance_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')
dm <- sg('get_distance_matrix', 'TEST')
