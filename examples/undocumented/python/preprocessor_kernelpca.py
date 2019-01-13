#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 0.01, 1.0], [data, 0.05, 2.0]]

def preprocessor_kernelpca (data, threshold, width):
	from shogun import RealFeatures
	from shogun import KernelPCA
	import shogun as sg
	import numpy as np

	features = RealFeatures(data)

	kernel = sg.kernel("GaussianKernel", log_width=np.log(width))
	kernel.init(features, features)

	preprocessor = KernelPCA(kernel)
	preprocessor.fit(features)
	preprocessor.set_target_dim(2)
	features = preprocessor.transform(features)

	return features


if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca(*parameter_list[0])

