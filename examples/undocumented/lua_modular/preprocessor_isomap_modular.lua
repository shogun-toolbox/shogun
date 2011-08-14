require 'shogun'
require 'load'

data = load_numbers('../data/fm_train_real.dat')

parameter_list = {{data}}

function preprocessor_isomap_modular(data)
	features = RealFeatures(data)
		
	preprocessor = Isomap()
	preprocessor:set_target_dim(1)
	preprocessor:apply_to_feature_matrix(features)

	return features
end

if debug.getinfo(3) == nill then
	print 'Isomap'
	preprocessor_isomap_modular(unpack(parameter_list[1]))
end

