# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_landmarkisomap_modular(data)
	
	features = RealFeatures(data)
		
	preprocessor = LandmarkIsomap()
	preprocessor.set_target_dim(1)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	print 'LandmarkIsomap'
	preprocessor_landmarkisomap_modular(*parameter_list[0])


end
