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

# Distance
print('Distance')

width=1.7
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('set_kernel', 'DISTANCE', size_cache, width)

dump <- sg('set_features', 'TRAIN', fm_train_real)
km=sg('get_kernel_matrix', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_real)
km=sg('get_kernel_matrix', 'TEST')
