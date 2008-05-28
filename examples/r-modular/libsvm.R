dyn.load('features/Features.so')
dyn.load('kernel/Kernel.so')
dyn.load('classifier/Classifier.so')
load('kernel/Kernel.RData')
cacheMetaData(1)
load('features/Features.RData')
cacheMetaData(1)
load('classifier/Classifier.RData')
cacheMetaData(1)

num <- 1000
dist <- 1
width <- 2.1
C <- 1
epsilon <- 1e-5
nthreads <- as.integer(1)

traindata_real <- matrix(c(rnorm(2*num)-dist,rnorm(2*num)+dist),2,2*num)
testdata_real <- matrix(c(rnorm(2*num)-dist,rnorm(2*num)+dist),2,2*num)

trainlab <- c(rep(-1,num), rep(1,num))
testlab <- c(rep(-1,num), rep(1,num))

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
kernel <- GaussianKernel(feats_train, feats_train, width)

labels <- Labels(trainlab)
svm <- LibSVM(C, kernel, labels)
svm$parallel$set_num_threads(svm$parallel, nthreads)
svm$set_epsilon(svm, epsilon)
svm$train()
kernel$init(kernel, feats_train, feats_test)
outlab <- svm$classify(svm)
out=outlab$get_labels(outlab)
testerr <- mean(sign(out)!=testlab)
print(testerr)
