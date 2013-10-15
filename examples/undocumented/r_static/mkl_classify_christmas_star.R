# This script should enable you to rerun the experiment in the
# paper that we labeled with "christmas star".
#
# The task is to classify two star-shaped classes that share the
# midpoint. The difficulty of the learning problem depends on the
# distance between the classes, which is varied
#
# Our model selection leads to a choice of C <- 0.5. The model
# selection is not repeated inside this script.
library(sg)

# Preliminary settings:

C <- 0.5         # SVM Parameter
cache_size <- 50 # cache per kernel in MB
svm_eps<-1e-3	 # svm epsilon
mkl_eps<-1e-3	 # mkl epsilon

no_obs <- 20   # number of observations / data points (sum for train and test and both classes)
k_star <- 20     # number of "leaves" of the stars
alpha <- 0.3     # noise level of the data

radius_star <- matrix(0, length(seq(4.1, 10, 0.2)), 2)
radius_star[,1] <- seq(4.1, 10, 0.2)                      # increasing radius of the 1.class
radius_star[,2] <- matrix(4, length(radius_star[,1]),1)   # fixed radius 2.class
                                           # distanz between the classes: diff(radius_star(:,1)-radius_star(:,2))
rbf_width <- c(0.01, 0.1, 1, 10, 1000)     # different width for the five used rbf kernels


####
#### Great loop: train MKL for every data set (the different distances between the stars)
####

sg('loglevel', 'ERROR')
sg('echo', 'OFF')

w = matrix(0, length(1:dim(radius_star)[1]), length(rbf_width))

result.trainout=matrix(0, length(1:dim(radius_star)[1]), 2*no_obs)
result.testout=matrix(0, length(1:dim(radius_star)[1]), 2*no_obs)
result.trainerr=matrix(0,length(rbf_width), 1)
result.testerr=matrix(0,length(rbf_width), 1)

for (kk in 1:dim(radius_star)[1]) {
  # data generation
  print(sprintf('MKL for radius %+02.2f                                                      ', radius_star[kk,1]))

  dummy <- matrix(0, 2, 4*no_obs)
  dummy[1,] <- runif(4*no_obs)
  noise <- alpha*rnorm(4*no_obs)

  dummy[2,] <- sin(k_star*pi*dummy[1,]) + noise         # sine
  dummy[2,1:(2*no_obs)] <- dummy[2,1:(2*no_obs)]+ radius_star[kk,1]         # distanz shift: first class
  dummy[2,(2*no_obs+1):dim(dummy)[2]] <- dummy[2,(2*no_obs+1):dim(dummy)[2]]+ radius_star[kk,2] # distanz shift: second class

  dummy[1,] <- 2*pi*dummy[1,]

  x <- matrix(0, dim(dummy)[1], dim(dummy)[2])
  x[1,] <-  dummy[2,]*sin(dummy[1,])
  x[2,] <-  dummy[2,]*cos(dummy[1,])

  train_y <- c(-matrix(1,1, no_obs), matrix(1,1,no_obs))
  test_y <- c(-matrix(1,1, no_obs), matrix(1,1,no_obs))

  train_x <- matrix(0, 0, seq(1,dim(x)[2]/2))
  train_x <- x[,seq(1,dim(x)[2],2)]
  test_x  <- x[,seq(2,dim(x)[2],2)]

  rm('dummy', 'x')

  # train MKL

  sg('clean_kernel')
  sg('clean_features', 'TRAIN')
  sg('add_features','TRAIN', train_x)       # set a trainingset for every SVM
  sg('add_features','TRAIN', train_x)
  sg('add_features','TRAIN', train_x)
  sg('add_features','TRAIN', train_x)
  sg('add_features','TRAIN', train_x)
  sg('set_labels','TRAIN', train_y)         # set the labels
  sg('new_classifier', 'MKL_CLASSIFICATION')
  sg('mkl_parameters', mkl_eps, 0)
  sg('svm_epsilon', svm_eps)
  sg('set_kernel', 'COMBINED', 0)
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[1])
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[2])
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[3])
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[4])
  sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width[5])
  sg('c', C)
  sg('train_classifier')
  alphas <- sg('get_svm')[2]
  w[kk,] <- sg('get_subkernel_weights')

  # calculate train error
  sg('clean_features', 'TEST')
  sg('add_features','TEST',train_x)
  sg('add_features','TEST',train_x)
  sg('add_features','TEST',train_x)
  sg('add_features','TEST',train_x)
  sg('add_features','TEST',train_x)
  sg('set_labels','TEST', train_y)
  sg('set_threshold', 0)

  result.trainout[kk,]<-sg('classify')
  result.trainerr[kk]  <- mean(train_y!=sign(result.trainout[kk,]))

  # calculate test error

  sg('clean_features', 'TEST')
  sg('add_features','TEST',test_x)
  sg('add_features','TEST',test_x)
  sg('add_features','TEST',test_x)
  sg('add_features','TEST',test_x)
  sg('add_features','TEST',test_x)
  sg('set_labels','TEST',test_y)
  sg('set_threshold', 0)
  result.testout[kk,]<-sg('classify')
  result.testerr[kk]  <- mean(test_y!=sign(result.testout[kk,]))
}
cat('done. now w contains the kernel weightings and result test/train outputs and errors')
