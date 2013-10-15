require 'modshogun'
require 'load'

data = load_numbers('../data/fm_train_real.dat')

parameter_list = {{data}}

function converter_isomap_modular(data)
	features = modshogun.RealFeatures(data)

	converter = modshogun.Isomap()
	converter:set_target_dim(1)
	converter:apply(features)

	return features
end

if debug.getinfo(3) == nill then
	print 'Isomap'
	converter_isomap_modular(unpack(parameter_list[1]))
end

