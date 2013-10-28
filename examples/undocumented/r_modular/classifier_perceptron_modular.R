library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# perceptron
print('Perceptron')

feats_train <- RealFeatures()
dump <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dump <- feats_test$set_feature_matrix(fm_test_real)

learn_rate <- 1.
max_iter <- as.integer(1000)
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

perceptron <- Perceptron(feats_train, labels)
dump <- perceptron$set_learn_rate(learn_rate)
dump <- perceptron$set_max_iter(max_iter)
dump <- perceptron$train()

dump <- perceptron$set_features(feats_test)
lab <- perceptron$apply()
out <- lab$get_labels()
