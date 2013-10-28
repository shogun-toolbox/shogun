library("sg")

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_multiclass <- as.double(as.matrix(read.table('../data/label_train_multiclass.dat')))

# KNN
print('KNN')
k <- 3

dump <- sg('set_features', 'TRAIN', fm_train_real)
dump <- sg('set_labels', 'TRAIN', label_train_multiclass)
dump <- sg('set_distance', 'EUCLIDEAN', 'REAL')
dump <- sg('new_classifier', 'KNN')
dump <- sg('train_classifier', k)

dump <- sg('set_features', 'TEST', fm_test_real)
result <- sg('classify')
