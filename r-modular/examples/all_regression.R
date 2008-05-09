dyn.load('features/Features.so')
dyn.load('kernel/Kernel.so')
dyn.load('regression/Regression.so')
load('features/Features.RData')
cacheMetaData(1)
load('kernel/Kernel.RData')
cacheMetaData(1)
load('regression/Regression.RData')
cacheMetaData(1)

#source('kernel/Kernel.R')
#source('features/Features.R')
#source('regression/Regression.R')
#cacheMetaData(1)

num <- 50 #number of example
len <- 10 #number of dimensions
dist <- 1.5

# Explicit examples on how to use regressions

traindata_real <- matrix(c(rnorm(2*num)-dist,rnorm(2*num)+dist),2,2*num)
testdata_real <- matrix(c(rnorm(2*num)-dist,rnorm(2*num)+dist),2,2*num)

###########################################################################
# svm-based
###########################################################################

# libsvm based support vector regression
print('SVRLight')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1;
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(3)
lab <- round(runif(feats_train$get_num_vectors()))*2-1
labels <- Labels(lab)

svr <- SVRLight(C, epsilon, kernel, labels)
svr$set_tube_epsilon(svr, tube_epsilon)
svr$parallel$set_num_threads(svr$parallel, num_threads)
svr$train()

kernel$init(kernel, feats_train, feats_test)
outlab <- svr$classify(svr)
out <- outlab$get_labels(outlab)

# libsvm based support vector regression
print('LibSVR')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(3)
lab <- round(runif(feats_train$get_num_vectors()))*2-1
labels <- Labels(lab)

svr <- LibSVR(C, epsilon, kernel, labels)
svr$set_tube_epsilon(svr, tube_epsilon)
svr$parallel$set_num_threads(svr$parallel, num_threads);
svr$train();

kernel$init(kernel, feats_train, feats_test);
outlab <- svr$classify(svr)
out <- outlab$get_labels(outlab);

############################################################################
# misc
############################################################################

# kernel ridge regression
print('KRR')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 0.8
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.42
tau <- 1e-6
num_threads <- as.integer(1)
lab <- round(runif(feats_train$get_num_vectors()))*2-1
labels <- Labels(lab)

krr <- KRR(tau, kernel, labels)
krr$parallel$set_num_threads(krr$parallel, num_threads)
krr$train()

kernel$init(kernel, feats_train, feats_test)
outlab <- krr$classify(krr)
out <- outlab$get_labels(outlab)
