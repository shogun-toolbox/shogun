#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 0.01, 1.0], [data, 0.05, 2.0]]

def preprocessor_kernelpca_modular (data, threshold, width):
	from modshogun import RealFeatures
	from modshogun import KernelPCA
	from modshogun import GaussianKernel
	
	features = RealFeatures(data)
	
	kernel = GaussianKernel(features,features,width)
		
	preprocessor = KernelPCA(kernel)
	preprocessor.init(features)
	preprocessor.set_target_dim(2)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca_modular(*parameter_list[0])

