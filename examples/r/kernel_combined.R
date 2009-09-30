# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

size_cache <- 10

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna.dat')))

# Combined
print('Combined')

dump <- sg('clean_features', 'TRAIN')
dump <- sg('clean_features', 'TEST')
dump <- sg('set_kernel', 'COMBINED', size_cache)
dump <- sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)
dump <- sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 3, FALSE)
dump <- sg('add_features', 'TRAIN', fm_train_real)
dump <- sg('add_features', 'TEST', fm_test_real)

km <- sg('get_kernel_matrix', 'TRAIN')

km <- sg('get_kernel_matrix', 'TEST')

