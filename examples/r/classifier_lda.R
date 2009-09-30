# Explicit examples on how to use the different classifiers
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

size_cache <- 10
C <- 10
epsilon <- 1e-5
use_bias <- TRUE

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna.dat')))
label_train_twoclass <- as.real(as.matrix(read.table('../data/label_train_twoclass.dat')))
label_train_multiclass <- as.real(as.matrix(read.table('../data/label_train_multiclass.dat')))

# LDA
print('LDA')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LDA')
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
result <- sg('classify')
