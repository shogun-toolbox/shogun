# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat],[traindat,testdat]]

def distance_sparseeuclidean_modular(fm_train_real=traindat,fm_test_real=testdat)

# *** 	realfeat=RealFeatures(fm_train_real)
	realfeat=Modshogun::RealFeatures.new
	realfeat.set_features(fm_train_real)
# *** 	feats_train=SparseRealFeatures()
	feats_train=Modshogun::SparseRealFeatures.new
	feats_train.set_features()
	feats_train.obtain_from_simple(realfeat)
# *** 	realfeat=RealFeatures(fm_test_real)
	realfeat=Modshogun::RealFeatures.new
	realfeat.set_features(fm_test_real)
# *** 	feats_test=SparseRealFeatures()
	feats_test=Modshogun::SparseRealFeatures.new
	feats_test.set_features()
	feats_test.obtain_from_simple(realfeat)

# *** 	distance=SparseEuclidianDistance(feats_train, feats_train)
	distance=Modshogun::SparseEuclidianDistance.new
	distance.set_features(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test


end
if __FILE__ == $0
	puts 'SparseEuclidianDistance'
	distance_sparseeuclidean_modular(*parameter_list[0])

end
