# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_hierarchical_modular(fm_train=traindat,merges=3)



# *** 	feats_train=RealFeatures(fm_train)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train)
# *** 	distance=EuclidianDistance(feats_train, feats_train)
	distance=Modshogun::EuclidianDistance.new
	distance.set_features(feats_train, feats_train)

# *** 	hierarchical=Hierarchical(merges, distance)
	hierarchical=Modshogun::Hierarchical.new
	hierarchical.set_features(merges, distance)
	hierarchical.train()

	out_distance = hierarchical.get_merge_distances()
	out_cluster = hierarchical.get_cluster_pairs()

	return hierarchical,out_distance,out_cluster 


end
if __FILE__ == $0
	puts 'Hierarchical'
	clustering_hierarchical_modular(*parameter_list[0])

end
