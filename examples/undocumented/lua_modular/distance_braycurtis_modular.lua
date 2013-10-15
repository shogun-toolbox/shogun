require 'modshogun'
require 'load'

traindat = load_numbers('../data/fm_train_real.dat')
testdat = load_numbers('../data/fm_test_real.dat')

parameter_list = {{traindat,testdat},{traindat,testdat}}

function distance_braycurtis_modular (fm_train_real,fm_test_real)

	feats_train=modshogun.RealFeatures(fm_train_real)
	feats_test=modshogun.RealFeatures(fm_test_real)

	distance=modshogun.BrayCurtisDistance(feats_train, feats_train)

	dm_train=distance:get_distance_matrix()
	distance:init(feats_train, feats_test)
	dm_test=distance:get_distance_matrix()

	return distance,dm_train,dm_test
end

if debug.getinfo(3) == nill then
	print 'BrayCurtisDistance'
	distance_braycurtis_modular(unpack(parameter_list[1]))
end
