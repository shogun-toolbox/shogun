# note shogun is 0 based, R is 1 based
dyn.load('features/Features.so')
source("features/Features.R")
cacheMetaData(1)

x=c(1.0, 2.0, 3.0)
lab <- Labels(x)
print(sprintf('labels: %f', lab$get_labels(lab)[1]))
print(sprintf('lab: %d', lab$get_num_labels()))


lab$set_label(lab, integer(0), 17)
print(sprintf('lab: %f', lab$get_label(lab, integer(0))))

dyn.load('kernel/Kernel.so')
source("kernel/Kernel.R")
cacheMetaData(1)

k <- GaussianKernel(10, 1.0)
print(sprintf('weight: %f', k$get_combined_kernel_weight()))

