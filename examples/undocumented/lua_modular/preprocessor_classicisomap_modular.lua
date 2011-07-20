require 'shogun'
require 'load'

data = load_numbers('../data/fm_train_real.dat')

parameter_list = {{data}}

function preprocessor_classicisomap_modular(data)
	features = RealFeatures(data)
		
	preprocessor = ClassicIsomap()
	preprocessor:set_target_dim(1)
	preprocessor:apply_to_feature_matrix(features)

	return features
end

print 'ClassicIsomap'
preprocessor_classicisomap_modular(unpack(parameter_list[1]))

