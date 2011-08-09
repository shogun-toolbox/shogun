# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_locallylinearembedding_modular(data,k)
	
	features = RealFeatures(data)
		
	preprocessor = LocallyLinearEmbedding()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	print 'LocallyLinearEmbedding'
	preprocessor_locallylinearembedding_modular(*parameter_list[0])


end
