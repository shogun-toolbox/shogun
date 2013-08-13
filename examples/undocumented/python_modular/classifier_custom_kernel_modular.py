#!/usr/bin/env python
parameter_list = [[1,7],[2,8]]

def classifier_custom_kernel_modular (C=1,dim=7):
	from modshogun import RealFeatures, BinaryLabels, CustomKernel, LibSVM
	from numpy import diag,ones,sign,rand,seed

	seed((C,dim))

	lab=sign(2*rand(dim) - 1)
	data=rand(dim, dim)
	symdata=data*data.T + diag(ones(dim))
    
	kernel=CustomKernel()
	kernel.set_full_kernel_matrix_from_full(data)
	labels=BinaryLabels(lab)
	svm=LibSVM(C, kernel, labels)
	svm.train()
	predictions =svm.apply() 
	out=svm.apply().get_labels()
	return svm,out

if __name__=='__main__':
	print('custom_kernel')
	classifier_custom_kernel_modular(*parameter_list[0])
