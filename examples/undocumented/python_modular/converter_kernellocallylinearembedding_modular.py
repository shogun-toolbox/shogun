#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_kernellocallylinearembedding_modular (data_fname,k):
	try:
		from modshogun import RealFeatures, KernelLocallyLinearEmbedding, LinearKernel, CSVFile

		features = RealFeatures(CSVFile(data_fname))

		kernel = LinearKernel()

		converter = KernelLocallyLinearEmbedding(kernel)
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.apply(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('KernelLocallyLinearEmbedding')
	converter_kernellocallylinearembedding_modular(*parameter_list[0])

