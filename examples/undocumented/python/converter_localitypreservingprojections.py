#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data,20],[data,30]]

def converter_localitypreservingprojections (data_fname,k):
	from shogun import RealFeatures, CSVFile
	from shogun import LocalityPreservingProjections

	features = RealFeatures(CSVFile(data_fname))
	converter = LocalityPreservingProjections()
	converter.set_target_dim(1)
	converter.set_k(k)
	converter.set_tau(2.0)
	converter.transform(features)

	return features

if __name__=='__main__':
	print('LocalityPreservingProjections')
	#converter_localitypreservingprojections(*parameter_list[0])

