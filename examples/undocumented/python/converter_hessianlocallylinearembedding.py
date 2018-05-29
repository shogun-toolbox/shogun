#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_hessianlocallylinearembedding (data_fname,k):
	try:
		from shogun import RealFeatures, CSVFile
		try:
			from shogun import HessianLocallyLinearEmbedding
		except ImportError:
			print("HessianLocallyLinearEmbedding not available")
			exit(0)

		features = RealFeatures(CSVFile(data))

		converter = HessianLocallyLinearEmbedding()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.transform(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('HessianLocallyLinearEmbedding')
	converter_hessianlocallylinearembedding(*parameter_list[0])

