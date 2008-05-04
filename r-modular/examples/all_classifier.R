dyn.load('features/Features.so')
dyn.load('classifier/Classifier.so')
dyn.load('kernel/Kernel.so')
dyn.load('distance/Distance.so')
source('features/Features.R')
source('classifier/Classifier.R')
source('kernel/Kernel.R')
source('distance/Distance.R')
cacheMetaData(1)

num <- 40
len <- 3
dist <- 2

# Explicit examples on how to use the different classifiers

acgt <- c('A', 'C', 'G', 'T')
traindata_dna <- list()
testdata_dna <- list()
for (i in 1:num)
{
	traindata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep="", collapse="")
	testdata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep="", collapse="")
}

traindata_dna=c(traindata_dna,recursive=TRUE)
testdata_dna=c(testdata_dna,recursive=TRUE)

trainlab <- c(rep(-1,num/2), rep(1,num/2))
testlab <- c(rep(-1,num/2), rep(1,num/2))
trainlab_multi <- c(rep(0,num/4), rep(1,num/4), rep(2,num/4), rep(3,num/4))
traindata_real <- matrix(c(rnorm(num)-dist,rnorm(num)+dist),2,num)
testdata_real <- matrix(c(rnorm(num)-dist,rnorm(num)+dist),2,num)

###########################################################################
# kernel-based SVMs
###########################################################################

# svm light
print('SVMLight')

feats_train <- StringCharFeatures("DNA")
feats_train$set_string_features(feats_train, traindata_dna)
feats_test <- StringCharFeatures("DNA")
feats_test$set_string_features(feats_test, testdata_dna)
degree <- 20

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(3)
labels <- Labels(trainlab)

svm <- SVMLight(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# libsvm
print('LibSVM')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(2)
labels <- Labels(trainlab)

svm <- LibSVM(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# gpbtsvm
print('GPBTSVM')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(2)
labels <- Labels(trainlab)

svm <- GPBTSVM(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# mpdsvm
print('MPDSVM')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(1)
labels <- Labels(trainlab)

svm <- MPDSVM(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# libsvmmulticlass
print('LibSVMMultiClass')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(8)
labels <- Labels(trainlab_multi)

svm <- LibSVMMultiClass(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# libsvm oneclass
print('LibSVMOneClass')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(4)

svm <- LibSVMOneClass(C, kernel)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# gmnpsvm
print('GMNPSVM')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(1)
labels <- Labels(trainlab_multi)

svm <- GMNPSVM(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# run with batch or linadd on LibSVM
###########################################################################

# batch & linadd
print('LibSVM batch')

feats_train <- StringCharFeatures("DNA")
feats_train$set_string_features(feats_train, traindata_dna)
feats_test <- StringCharFeatures("DNA")
feats_test$set_string_features(feats_test, testdata_dna)
degree <- 20

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- 2
labels <- Labels(trainlab)

svm <- LibSVM(C, kernel, labels)
svm$set_epsilon(svm, epsilon)
svm$set_tube_epsilon(svm, tube_epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$train()

kernel$init(kernel, feats_train, feats_test)

print(sprintf('LibSVM Objective: %f num_sv: %d', svm$get_objective(), svm$get_num_support_vectors()))
svm$set_batch_computation_enabled(svm, FALSE)
svm$set_linadd_enabled(svm, FALSE)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

svm$set_batch_computation_enabled(svm, TRUE)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# linear SVMs
###########################################################################

# subgradient based svm
print('SubGradientSVM')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-3
num_threads <- as.integer(1)
max_train_time <- 1.
labels <- Labels(trainlab)

svm <- SubGradientSVM(C, feats_train, labels)
svm$set_epsilon(svm, epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$set_bias_enabled(svm, FALSE)
svm$set_max_train_time(svm, max_train_time)
svm$train()

svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# svm ocas
print('SVMOcas')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(trainlab)

svm <- SVMOcas(C, feats_train, labels)
svm$set_epsilon(svm, epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$set_bias_enabled(svm, FALSE)
svm$train()

svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# sgd
print('SVMSGD')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(trainlab)

svm <- SVMSGD(C, feats_test, labels)
svm$io$set_loglevel(svm$io, 0)
svm$train()

svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# liblinear
print('LibLinear')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(trainlab)

svm <- LibLinear(C, feats_train, labels)
svm$set_epsilon(svm, epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$set_bias_enabled(svm, TRUE)
svm$train()

svm$set_features(svm, feats_test)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

# svm lin
print('SVMLin')

realfeat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, realfeat)
realfeat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, realfeat)

C <- 0.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- Labels(trainlab)

svm <- SVMLin(C, feats_train, labels)
svm$set_epsilon(svm, epsilon)
svm$parallel$set_num_threads(svm$parallel, num_threads)
svm$set_bias_enabled(svm, TRUE)
svm$train()

svm$set_features(svm, feats_test)
svm$get_bias(svm)
svm$get_w(svm)
lab <- svm$classify(svm)
out <- lab$get_labels(lab)

###########################################################################
# misc classifiers
###########################################################################

# perceptron
print('Perceptron')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(traindata_real)

learn_rate <- 1.
max_iter <- as.integer(1000)
num_threads <- as.integer(1)
labels <- Labels(trainlab)

perceptron <- Perceptron(feats_train, labels)
perceptron$set_learn_rate(perceptron, learn_rate)
perceptron$set_max_iter(perceptron, max_iter)
perceptron$train()

perceptron$set_features(perceptron, feats_test)
lab <- perceptron$classify(perceptron)
out <- lab$get_labels(lab)

# knn
print('KNN')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
distance <- EuclidianDistance()

k <- as.integer(3)
num_threads <- as.integer(1)
labels <- Labels(trainlab)

knn <- KNN(k, distance, labels)
knn$parallel$set_num_threads(knn$parallel, num_threads)
knn$train()

distance$init(distance, feats_train, feats_test)
lab <- knn$classify(knn)
out <- lab$get_labels(lab)

# lda
print('LDA')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)

gamma <- 3
num_threads <- as.integer(1)
labels <- Labels(trainlab)

lda <- LDA(gamma, feats_train, labels)
lda$parallel$set_num_threads(lda$parallel, num_threads)
lda$train()

lda$get_bias(lda)
lda$get_w(lda)
lda$set_features(lda, feats_test)
lab <- lda$classify(lda)
out <- lab$get_labels(lab)
