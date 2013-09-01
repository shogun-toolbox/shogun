require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat],[traindat,testdat]]

def distance_canberra_modular(fm_train_real=traindat,fm_test_real=testdat)

	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)

	distance=Modshogun::CanberraMetric.new(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

end

if __FILE__ == $0
	puts 'CanberaMetric'
	pp distance_canberra_modular(*parameter_list[0])
end
