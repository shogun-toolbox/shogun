# Explicit examples on how to use the different preprocs
#
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

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true
use_sign <- FALSE
normalization <- 'FULL'

# Comm Ulong String
print('CommUlongString')

dump <- sg('add_preproc', 'SORTULONGSTRING')
dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')

dump <- sg('set_features', 'TEST', fm_test_dna, 'DNA')
dump <- sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TEST')

dump <- sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization)
km <- sg('get_kernel_matrix', 'TRAIN')
km <- sg('get_kernel_matrix', 'TEST')
