# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_hessianlocallylinearembedding_modular(data,k)
	
	features = RealFeatures(data)
		
	preprocessor = HessianLocallyLinearEmbedding()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	puts 'HessianLocallyLinearEmbedding'
	preprocessor_hessianlocallylinearembedding_modular(*parameter_list[0])


end
