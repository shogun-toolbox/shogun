#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data]]

def converter_isomap (data_fname):
	from shogun import RealFeatures, CSVFile
	from shogun import Isomap
		
	features = RealFeatures(CSVFile(data))

	converter = Isomap()
	converter.set_k(20)
	converter.set_target_dim(1)
	converter.transform(features)

	return features

if __name__=='__main__':
	print('Isomap')
	#converter_isomap(*parameter_list[0])

