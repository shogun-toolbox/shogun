# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_laplacianeigenmaps_modular(data,k)
	
	features = RealFeatures(data)
		
	preprocessor = LaplacianEigenmaps()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.set_tau(2.0)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	print 'LaplacianEigenmaps'
	preprocessor_laplacianeigenmaps_modular(*parameter_list[0])


end
