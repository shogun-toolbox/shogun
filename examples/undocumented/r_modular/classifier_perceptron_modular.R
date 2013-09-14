library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat')$V1)

# perceptron
print('Perceptron')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

learn_rate <- 1.
max_iter <- as.integer(1000)
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

perceptron <- Perceptron(feats_train, labels)
dump <- perceptron$set_learn_rate(perceptron, learn_rate)
dump <- perceptron$set_max_iter(perceptron, max_iter)
dump <- perceptron$train(perceptron)

dump <- perceptron$set_features(perceptron, feats_test)
lab <- perceptron$apply(perceptron)
out <- lab$get_labels(lab)
