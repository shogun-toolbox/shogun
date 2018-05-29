#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_localtangentspacealignment (data_fname,k):
	try:
		from shogun import RealFeatures, CSVFile
		try:
			from shogun import LocalTangentSpaceAlignment
		except ImportError:
			print("LocalTangentSpaceAlignment not available")
			exit(0)
			
		features = RealFeatures(CSVFile(data_fname))

		converter = LocalTangentSpaceAlignment()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.transform(features)

		return features
	except ImportError:
		print('No Eigen3 available')


if __name__=='__main__':
	print('LocalTangentSpaceAlignment')
	converter_localtangentspacealignment(*parameter_list[0])

