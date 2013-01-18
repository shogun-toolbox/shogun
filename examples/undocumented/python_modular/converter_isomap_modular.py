#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_isomap_modular (data):
	try:
		from shogun.Features import RealFeatures
		from shogun.Converter import Isomap
		
		features = RealFeatures(data)
			
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

