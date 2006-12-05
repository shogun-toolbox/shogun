# This script should enable you to rerun the experiment in the
# paper that we labeled "mixture linear and sine ".
#
# The task is to learn a regression function where the true function
# is given by a mixture of 2 sine waves in addition to a linear trend. 
# We vary the frequency of the second higher frequency sine wave. 

# Setup: MKL on 10 RBF kernels of different widths on 1000 examples

#  source("/fml/ag-raetsch/home/fabio/projects/shogun/R/sg/tests/not_yet/mkl_lin_sine_mix.R")

#load  shogun
library(sg)

# kernel width for 10 basic SVMs
rbf_width <- array(0.0, dim=c(1,10))
rbf_width[1] = 0.001
rbf_width[2] = 0.005
rbf_width[3] = 0.01
rbf_width[4] = 0.05
rbf_width[5] = 0.1
rbf_width[6] = 1
rbf_width[7] = 10
rbf_width[8] = 50
rbf_width[9] = 100
rbf_width[10] = 1000

# SVM parameter
C          = 1
cache_size = 50
mkl_eps    = 1e-4
svm_eps    = 1e-4
svm_tube   = 0.01
debug = 0

# data
f <- c(0:20)  # parameter that varies the frequency of the second sine wave
no_obs = 1000    # number of observations

send_command("loglevel ALL");
send_command("echo ON");

weights = array(dim=c(21,10))

stepsize = (4*pi)/(no_obs-1)
train_x = c(0:(no_obs-1))
for (i in c(1:no_obs)) {
   train_x[i] = train_x[i] * stepsize
}

trend = 2 * train_x* ((pi)/(max(train_x)-min(train_x)))
wave1 = sin(train_x)
wave2 = sin(f[kk]*train_x)
train_y = trend + wave1 + wave2

for (kk in c(1:length(f))) {  #Big loop
  
  #data generation
   wave1 = sin(train_x)
   wave2 = sin(f[kk]*train_x)
   train_y = trend + wave1 + wave2

  #MKL learning
   send_command("new_svm SVRLIGHT")
   send_command("use_mkl 1")                      
   send_command("use_precompute 0")      #precompute every SINGLE kernel!
   send_command(sprintf("mkl_parameters %f 0",mkl_eps))
   send_command(sprintf("c %f",C))                
   send_command(sprintf("svm_epsilon %f",svm_eps))
   send_command(sprintf("svr_tube_epsilon %f",svm_tube))
   send_command("clean_features TRAIN" )
   send_command("clean_kernels" )

   set_labels("TRAIN", train_y)              #set labels
   add_features("TRAIN", train_x)             #add features for every basic SVM
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)            
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   add_features("TRAIN", train_x)
   send_command("set_kernel COMBINED 0")
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[1]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[2]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[3]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[4]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[5]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[6]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[7]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[8]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[9]))
   send_command(sprintf("add_kernel 1 GAUSSIAN REAL %d %f", cache_size, rbf_width[10]))
   send_command("init_kernel TRAIN") 
   send_command("svm_train")

   weights[kk,] = get_subkernel_weights()
   cat("frequency:", f[kk], " rbf-kernel-weights: ", weights[kk,], "\n")
}
