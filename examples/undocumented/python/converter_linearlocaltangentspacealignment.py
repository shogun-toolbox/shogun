#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_linearlocaltangentspacealignment (data_fname,k):
	try:
		try:
			from shogun import LinearLocalTangentSpaceAlignment
		except ImportError:
			print("LinearLocalTangentSpaceAlignment not available")
			exit(0)
		import shogun as sg
			
		features = sg.create_features(sg.create_csv(data_fname))

		converter = LinearLocalTangentSpaceAlignment()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.transform(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('LinearLocalTangentSpaceAlignment')
	converter_linearlocaltangentspacealignment(*parameter_list[0])

