from numpy import *
from numpy.random import rand
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import CustomKernel
from shogun.Classifier import LibSVM

C=1
dim=7

lab=sign(2*rand(dim) - 1)
data=rand(dim, dim)
symdata=data+data.T

kernel=CustomKernel()
kernel.set_full_kernel_matrix_from_full(data)
labels=Labels(lab)
svm=LibSVM(C, kernel, labels)
svm.train()
out=svm.classify().get_labels()
