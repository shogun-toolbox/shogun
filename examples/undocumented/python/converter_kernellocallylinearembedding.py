#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_kernellocallylinearembedding (data_fname,k):
	try:
		from shogun import RealFeatures, LinearKernel, CSVFile
		try:
			from shogun import KernelLocallyLinearEmbedding
		except ImportError:
			print("KernelLocallyLinearEmbedding not available")
			exit(0)
			
		features = RealFeatures(CSVFile(data_fname))

		kernel = LinearKernel()

		converter = KernelLocallyLinearEmbedding(kernel)
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.transform(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('KernelLocallyLinearEmbedding')
	converter_kernellocallylinearembedding(*parameter_list[0])

