library("sg")

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(as.matrix(read.table('../data/label_train_twoclass.dat')))

# LDA
print('LDA')

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_twoclass)
dump <- sg('new_classifier', 'LDA')
dump <- sg('train_classifier')

dump <- sg('set_features', 'TEST', fm_test_real)
result <- sg('classify')
