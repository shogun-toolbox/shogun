library("sg")

size_cache <- 10
C <- 10
epsilon <- 1e-5
use_bias <- TRUE

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
label_train_twoclass <- as.double(as.matrix(read.table('../data/label_train_twoclass.dat')))

# Perceptron
print('Perceptron')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'PERCEPTRON')
# often does not converge
#dump <- sg('train_classifier')

#dump <- sg('set_features', 'TEST', fm_test_real)
#result <- sg('classify')
