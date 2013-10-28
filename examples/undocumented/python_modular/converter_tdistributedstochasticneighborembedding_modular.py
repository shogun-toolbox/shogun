#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data]]

def converter_tdistributedstochasticneighborembedding_modular(data_fname, seed=1):
	try:
		from modshogun import RealFeatures, TDistributedStochasticNeighborEmbedding
		from modshogun import Math_init_random, CSVFile

		# reproducible results
		Math_init_random(seed)
		features = RealFeatures(CSVFile(data_fname))

		converter = TDistributedStochasticNeighborEmbedding()
		converter.set_target_dim(2)

		embedding = converter.apply(features)

		return embedding
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('TDistributedStochasticNeighborEmbedding')
	converter_tdistributedstochasticneighborembedding_modular(*parameter_list[0])
