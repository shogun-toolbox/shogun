#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_localitypreservingprojections_modular (data_fname,k):
	try:
		from modshogun import RealFeatures, LocalityPreservingProjections, CSVFile
		
		features = RealFeatures(CSVFile(data_fname))
		converter = LocalityPreservingProjections()
		converter.set_target_dim(1)
		converter.set_k(k)
		converter.set_tau(2.0)
		converter.apply(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('LocalityPreservingProjections')
	converter_localitypreservingprojections_modular(*parameter_list[0])

