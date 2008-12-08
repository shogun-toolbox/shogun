library(shogun)

# Explicit examples on how to use the different classifiers

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(read.table('../data/label_train_dna42.dat'))
label_train_twoclass <- as.real(read.table('../data/label_train_twoclass.dat'))
label_train_multiclass <- as.real(read.table('../data/label_train_multiclass.dat'))

###########################################################################
# kernel-based SVMs
###########################################################################

# svm light
dosvmlight <- function()
{
	print('SVMLight')

	feats_train <- StringCharFeatures("DNA")
	dump <- feats_train$set_string_features(feats_train, fm_train_dna)
	feats_test <- StringCharFeatures("DNA")
	dump <- feats_test$set_string_features(feats_test, fm_test_dna)
	degree <- as.integer(20)

	kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C <- 0.017
	epsilon <- 1e-5
	tube_epsilon <- 1e-2
	num_threads <- as.integer(3)
	labels <- Labels(as.real(label_train_dna))

	svm <- SVMLight(C, kernel, labels)
	dump <- svm$set_epsilon(svm, epsilon)
	dump <- svm$set_tube_epsilon(svm, tube_epsilon)
	dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
	dump <- svm$train()

	dump <- kernel$init(kernel, feats_train, feats_test)
	lab <- svm$classify(svm)
	out <- lab$get_labels(lab)
}
try(dosvmlight())


# libsvm
print('LibSVM')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(2)
labels <- Labels(label_train_twoclass)

svm <- LibSVM(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# gpbtsvm
print('GPBTSVM')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(2)
labels <- Labels(label_train_twoclass)

svm <- GPBTSVM(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# mpdsvm
print('MPDSVM')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- MPDSVM(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# libsvmmulticlass
print('LibSVMMultiClass')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(8)
labels <- Labels(label_train_multiclass)

svm <- LibSVMMultiClass(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# libsvm twoclass
print('LibSVMOneClass')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(4)

svm <- LibSVMOneClass(C, kernel)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# gmnpsvm
print('GMNPSVM')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(1)
labels <- Labels(label_train_multiclass)

svm <- GMNPSVM(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# run with batch or linadd on LibSVM
###########################################################################

# batch & linadd
print('LibSVM batch')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(20)

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(2)
labels <- Labels(label_train_dna)

svm <- LibSVM(C, kernel, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$set_tube_epsilon(svm, tube_epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$train()

dump <- kernel$init(kernel, feats_train, feats_test)

#print(sprintf('LibSVM Objective: %f num_sv: %d', svm$get_objective(), svm$get_num_support_vectors()))
dump <- svm$set_batch_computation_enabled(svm, FALSE)
dump <- svm$set_linadd_enabled(svm, FALSE)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

dump <- svm$set_batch_computation_enabled(svm, TRUE)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# linear SVMs
###########################################################################

# subgradient based svm
print('SubGradientSVM')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-3
num_threads <- as.integer(1)
max_train_time <- 1.
labels <- Labels(label_train_twoclass)

svm <- SubGradientSVM(C, feats_train, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_bias_enabled(svm, FALSE)
dump <- svm$set_max_train_time(svm, max_train_time)
dump <- svm$train()

dump <- svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# svm ocas
print('SVMOcas')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
dump <- feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- SVMOcas(C, feats_train, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_bias_enabled(svm, FALSE)
dump <- svm$train()

dump <- svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# sgd
print('SVMSGD')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- SVMSGD(C, feats_train, labels)
#dump <- svm$io$set_loglevel(svm$io, 0)
dump <- svm$train()

dump <- svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# liblinear
print('LibLinear')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- LibLinear(C, feats_train, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_bias_enabled(svm, TRUE)
dump <- svm$train()

dump <- svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# svm lin
print('SVMLin')

realfeat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

svm <- SVMLin(C, feats_train, labels)
dump <- svm$set_epsilon(svm, epsilon)
dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
dump <- svm$set_bias_enabled(svm, TRUE)
dump <- svm$train()

dump <- svm$set_features(svm, feats_test)
dump <- svm$get_bias(svm)
dump <- svm$get_w(svm)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# misc classifiers
###########################################################################

# perceptron
print('Perceptron')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_train_real)

learn_rate <- 1.
max_iter <- as.integer(1000)
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

perceptron <- Perceptron(feats_train, labels)
dump <- perceptron$set_learn_rate(perceptron, learn_rate)
dump <- perceptron$set_max_iter(perceptron, max_iter)
dump <- perceptron$train()

dump <- perceptron$set_features(perceptron, feats_test)
lab <- perceptron$classify(perceptron)
out <- lab$get_labels(lab)

# knn
print('KNN')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
distance <- EuclidianDistance()

k <- as.integer(3)
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

knn <- KNN(k, distance, labels)
dump <- knn$parallel$set_num_threads(knn$parallel, num_threads)
dump <- knn$train()

dump <- distance$init(distance, feats_train, feats_test)
lab <- knn$classify(knn)
out <- lab$get_labels(lab)

# lda
print('LDA')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

gamma <- 3
num_threads <- as.integer(1)
labels <- Labels(label_train_twoclass)

lda <- LDA(gamma, feats_train, labels)
dump <- lda$parallel$set_num_threads(lda$parallel, num_threads)
dump <- lda$train()

dump <- lda$get_bias(lda)
dump <- lda$get_w(lda)
dump <- lda$set_features(lda, feats_test)
lab <- lda$classify(lda)
out <- lab$get_labels(lab)
