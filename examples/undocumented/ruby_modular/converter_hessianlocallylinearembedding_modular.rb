require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def converter_hessianlocallylinearembedding_modular(data,k)
	
	features = Modshogun::RealFeatures.new
	features.set_feature_matrix(data)
		
	preprocessor = Modshogun::HessianLocallyLinearEmbedding.new
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply(features)

	return features

end

if __FILE__ == $0
	puts 'HessianLocallyLinearEmbedding'
	pp converter_hessianlocallylinearembedding_modular(*parameter_list[0])
end
