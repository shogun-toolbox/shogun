from tools.load import LoadMatrix
lm = LoadMatrix()

data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 12]]

def converter_stochasticproximityembedding_modular(data, k):
	from shogun.Features import RealFeatures
	from shogun.Converter import StochasticProximityEmbedding, SPE_GLOBAL, SPE_LOCAL
	
	features = RealFeatures(data)
		
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

if __name__=='__main__':
	print('StochasticProximityEmbedding')
	converter_stochasticproximityembedding_modular(*parameter_list[0])
