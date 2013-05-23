#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_tdistributedstochasticneighborembedding_modular(data, seed=1):
	try:
		from shogun.Features import RealFeatures
		from shogun.Converter import TDistributedStochasticNeighborEmbedding
		from shogun.Mathematics import Math_init_random
		
		# reproducible results
		Math_init_random(seed)
		random.seed(seed)	

		features = RealFeatures(data)
			
		converter = TDistributedStochasticNeighborEmbedding()
		converter.set_target_dim(2)
		
		embedding = converter.apply(features)

		return embedding
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('TDistributedStochasticNeighborEmbedding')
	converter_tdistributedstochasticneighborembedding_modular(*parameter_list[0])
