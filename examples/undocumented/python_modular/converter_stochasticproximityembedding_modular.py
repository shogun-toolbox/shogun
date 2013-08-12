#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data, 20]]

def converter_stochasticproximityembedding_modular (data_fname, k):
	try:
		from modshogun import RealFeatures,StochasticProximityEmbedding, SPE_GLOBAL, SPE_LOCAL, CSVFile
		
		features = RealFeatures(CSVFile(data_fname))
			
		converter = StochasticProximityEmbedding()
		converter.set_target_dim(1)
		converter.set_nupdates(40)
		# Embed with local strategy
		converter.set_k(k)
		converter.set_strategy(SPE_LOCAL)
		converter.embed(features)
		# Embed with global strategy
		converter.set_strategy(SPE_GLOBAL)
		converter.embed(features)

		return features
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('StochasticProximityEmbedding')
	converter_stochasticproximityembedding_modular(*parameter_list[0])
