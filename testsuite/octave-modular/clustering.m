function y = clustering(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/clustering');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end
	if ~set_distance()
		return;
	end

	if strcmp(name, 'KMeans')==1
		clustering=KMeans(clustering_k, distance);
		clustering.train();

		radi=clustering.get_radiuses();
		radi=max(max(abs(radi-clustering_radi)));
		centers=clustering.get_cluster_centers();
		centers=max(max(abs(centers-clustering_centers)));

		data={'kmeans', radi, centers};
		y=check_accuracy(clustering_accuracy, data);

	elseif strcmp(name, 'Hierarchical')==1
		clustering=Hierarchical(clustering_merges, distance);
		clustering.train();

		merge_distances=clustering.get_merge_distances();
		merge_distances=max(max(abs(
			merge_distances-clustering_merge_distance)));
		pairs=clustering.get_cluster_pairs();
		pairs=max(max(abs(pairs-clustering_pairs)));

		data={'hierarchical', merge_distances, pairs};
		y=check_accuracy(clustering_accuracy, data);
	else
		error('Unsupported clustering %s', name);
	end
