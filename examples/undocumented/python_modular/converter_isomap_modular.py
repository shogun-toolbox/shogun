#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data]]

def converter_isomap_modular (data_fname):
	try:
		from modshogun import RealFeatures, Isomap, CSVFile
		
		features = RealFeatures(CSVFile(data))
			
		converter = Isomap()
		converter.set_k(20)
		converter.set_target_dim(1)
		converter.apply(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('Isomap')
	converter_isomap_modular(*parameter_list[0])

