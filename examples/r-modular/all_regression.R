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

# Explicit examples on how to use regressions

fm_train <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test <- as.matrix(read.table('../data/fm_test_real.dat'))
label_train <- as.real(as.matrix(read.table('../data/label_train_twoclass.dat')))

###########################################################################
# svm-based
###########################################################################

# libsvm based support vector regression
dosvrlight <- function()
{
	print('SVRLight')

	feats_train <- RealFeatures(fm_train)
	feats_test <- RealFeatures(fm_test)
	width <- 2.1;
	kernel <- GaussianKernel(feats_train, feats_train, width)

	C <- 0.017
	epsilon <- 1e-5
	tube_epsilon <- 1e-2
	num_threads <- as.integer(3)
	labels <- Labels(label_train)

	svr <- SVRLight(C, epsilon, kernel, labels)
	svr$set_tube_epsilon(svr, tube_epsilon)
	svr$parallel$set_num_threads(svr$parallel, num_threads)
	svr$train()

	kernel$init(kernel, feats_train, feats_test)
	outlab <- svr$classify(svr)
	out <- outlab$get_labels(outlab)
}
try(dosvrlight())


# libsvm based support vector regression
print('LibSVR')

feats_train <- RealFeatures(fm_train)
feats_test <- RealFeatures(fm_test)
width <- 2.1
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.017
epsilon <- 1e-5
tube_epsilon <- 1e-2
num_threads <- as.integer(3)
labels <- Labels(label_train)

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

feats_train <- RealFeatures(fm_train)
feats_test <- RealFeatures(fm_test)
width <- 0.8
kernel <- GaussianKernel(feats_train, feats_train, width)

C <- 0.42
tau <- 1e-6
num_threads <- as.integer(1)
labels <- Labels(label_train)

krr <- KRR(tau, kernel, labels)
krr$parallel$set_num_threads(krr$parallel, num_threads)
krr$train()

kernel$init(kernel, feats_train, feats_test)
outlab <- krr$classify(krr)
out <- outlab$get_labels(outlab)
