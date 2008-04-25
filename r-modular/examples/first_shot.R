dyn.load('features/Features.so')
source("features/Features.R")
cacheMetaData(1)

x=c(1.0, 2.0, 3.0)
lab <- Labels(x)
sprintf('labels: %f', lab$get_labels()[0])
sprintf('lab: %d', lab$get_num_labels())


#lab <- lab$set_label(0, 17.0)
#sprintf('lab: %f', lab$get_label(0))

dyn.load('kernel/Kernel.so')
source("kernel/Kernel.R")
cacheMetaData(1)

k <- GaussianKernel(10, 1.0)
sprintf('weight: %f', k$get_combined_kernel_weight())

