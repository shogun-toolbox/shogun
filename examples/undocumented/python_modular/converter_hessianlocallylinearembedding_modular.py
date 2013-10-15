#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_hessianlocallylinearembedding_modular (data_fname,k):
	try:
		from modshogun import RealFeatures, HessianLocallyLinearEmbedding, CSVFile

		features = RealFeatures(CSVFile(data))

		converter = HessianLocallyLinearEmbedding()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.apply(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('HessianLocallyLinearEmbedding')
	converter_hessianlocallylinearembedding_modular(*parameter_list[0])

