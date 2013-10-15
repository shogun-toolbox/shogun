# This script should enable you to rerun the experiment in the
# paper that we labeled "mixture linear and sine ".
#
# The task is to learn a regression function where the true function
# is given by a mixture of 2 sine waves in addition to a linear trend.
# We vary the frequency of the second higher frequency sine wave.

# Setup: MKL on 10 RBF kernels of different widths on 1000 examples

#load  shogun
library(sg)

# kernel width for 10 basic SVMs
rbf_width <- array(0.0, dim<-c(1,10))
rbf_width[1] <- 0.001
rbf_width[2] <- 0.005
rbf_width[3] <- 0.01
rbf_width[4] <- 0.05
rbf_width[5] <- 0.1
rbf_width[6] <- 1
rbf_width[7] <- 10
rbf_width[8] <- 50
rbf_width[9] <- 100
rbf_width[10] <- 1000

# SVM parameter
C          <- 1
cache_size <- 50
mkl_eps    <- 1e-4
svm_eps    <- 1e-4
svm_tube   <- 0.01
debug <- 0

# data
f <- c(0:20)  # parameter that varies the frequency of the second sine wave

#sg('loglevel', 'ALL')
#sg('echo', 'ON')

weights <- array(dim<-c(21,10))

no_obs <- 10    # number of observations
stepsize <- (4*pi)/(no_obs-1)
train_x <- c(0:(no_obs-1))
for (i in c(1:no_obs)) {
   train_x[i] <- train_x[i] * stepsize
}

trend <- 2 * train_x* ((pi)/(max(train_x)-min(train_x)))
wave1 <- sin(train_x)
wave2 <- sin(f[1]*train_x)
train_y <- trend + wave1 + wave2

train_x<-matrix(train_x,1, length(train_x))

weights=matrix(0, length(f), length(rbf_width))

for (kk in c(1:length(f))) {  #Big loop

   #data generation
   wave1 <- sin(train_x)
   wave2 <- sin(f[kk]*train_x)
   train_y <- trend + wave1 + wave2

   #MK Learning
   sg('new_classifier', 'MKL_REGRESSION')
   sg('mkl_parameters', mkl_eps, 0)
   sg('c', C)
   sg('svm_epsilon', svm_eps)
   sg('svr_tube_epsilon', svm_tube)
   sg('clean_features', 'TRAIN')
   sg('clean_kernel')

   sg('set_labels', 'TRAIN', train_y)              #set labels
   sg('add_features', 'TRAIN', train_x)             #add features for every basic SVM
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('add_features', 'TRAIN', train_x)
   sg('set_kernel', 'COMBINED', 0)
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[1])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[2])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[3])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[4])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[5])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[6])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[7])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[8])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[9])
   sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[10])
   sg('train_classifier')

   weights[kk,] <- sg('get_subkernel_weights')
   cat("frequency:", f[kk], " rbf-kernel-weights: ", weights[kk,], "\n")
}
