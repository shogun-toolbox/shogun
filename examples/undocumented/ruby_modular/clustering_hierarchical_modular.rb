require 'rubygems'
require 'modshogun'
require 'pp'
require 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_hierarchical_modular(fm_train=traindat,merges=3)

	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train)
	distance=Modshogun::EuclideanDistance.new(feats_train, feats_train)

	hierarchical=Modshogun::Hierarchical.new(merges, distance)
	hierarchical.train()

	out_distance = hierarchical.get_merge_distances()
	out_cluster = hierarchical.get_cluster_pairs()

	return hierarchical,out_distance,out_cluster

end

if __FILE__ == $0
	puts 'Hierarchical'
	pp clustering_hierarchical_modular(*parameter_list[0])
end
