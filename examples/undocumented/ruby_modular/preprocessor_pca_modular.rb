# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_pca_modular(data)
	
	features = RealFeatures(data)
		
	preprocessor = PCA()
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	puts 'PCA'
	preprocessor_pca_modular(*parameter_list[0])


end
