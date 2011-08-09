# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_hierarchical_modular(fm_train=traindat,merges=3)



	feats_train=RealFeatures(fm_train)
	distance=EuclidianDistance(feats_train, feats_train)

	hierarchical=Hierarchical(merges, distance)
	hierarchical.train()

	out_distance = hierarchical.get_merge_distances()
	out_cluster = hierarchical.get_cluster_pairs()

	return hierarchical,out_distance,out_cluster 


end
if __FILE__ == $0
	print 'Hierarchical'
	clustering_hierarchical_modular(*parameter_list[0])

end
