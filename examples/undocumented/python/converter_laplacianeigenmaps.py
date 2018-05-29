#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_laplacianeigenmaps (data_fname,k):
	try:
		from shogun import RealFeatures, CSVFile
		try:
			from shogun import LaplacianEigenmaps
		except ImportError:
			print("LaplacianEigenmaps not available")
			exit(0)
			
		features = RealFeatures(CSVFile(data_fname))

		converter = LaplacianEigenmaps()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.set_tau(20.0)
		converter.transform(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('LaplacianEigenmaps')
	converter_laplacianeigenmaps(*parameter_list[0])

