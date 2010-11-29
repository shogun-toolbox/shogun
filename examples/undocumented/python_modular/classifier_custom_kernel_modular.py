from numpy import *
from numpy.random import rand
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import CustomKernel
from shogun.Classifier import LibSVM
parameter_list = [[1,7],[2,8]]

def classifier_custom_kernel_modular(C=1,dim=7):
    lab=sign(2*rand(dim) - 1)
    data=rand(dim, dim)
    symdata=data*data.T
    
    kernel=CustomKernel()
    kernel.set_full_kernel_matrix_from_full(data)
    labels=Labels(lab)
    svm=LibSVM(C, kernel, labels)
    svm.train()
    predictions =svm.classify() 
    out=svm.classify().get_labels()
    return svm,out

if __name__=='__main__':
	print 'custom_kernel'
	classifier_custom_kernel_modular(*parameter_list[0])
