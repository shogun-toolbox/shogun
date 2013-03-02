#!/usr/bin/env python
from tools.load import LoadMatrix
import numpy

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_factoranalysis_modular(data):
	try:
		from shogun.Features import RealFeatures
		from shogun.Converter import FactorAnalysis
		from shogun.Distance import EuclideanDistance
		
		features = RealFeatures(data)
			
		converter = FactorAnalysis()
		converter.set_target_dim(2)
		embedding = converter.apply(features)

		return embedding
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('Factor Analysis')
	converter_factoranalysis_modular(*parameter_list[0])
