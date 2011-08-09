# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,3],[traindat,testdat,4]]

def distance_minkowski_modular(fm_train_real=traindat,fm_test_real=testdat,k=3)


	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=MinkowskiMetric(feats_train, feats_train, k)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test


end
if __FILE__ == $0
	print 'MinkowskiMetric'
	distance_minkowski_modular(*parameter_list[0])


end
