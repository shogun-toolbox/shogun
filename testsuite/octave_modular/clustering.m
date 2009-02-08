function y = clustering(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/clustering');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	Math_init_random(init_random);
	rand('state', init_random);

	if ~set_features('distance_')
		return;
	end
	if ~set_distance()
		return;
	end

	if strcmp(clustering_name, 'KMeans')==1
		clustering=KMeans(clustering_k, distance);
		clustering.train();

		radi=clustering.get_radiuses();
		radi=max(max(abs(radi-clustering_radi)));
		centers=clustering.get_cluster_centers();
		centers=max(max(abs(centers-clustering_centers)));

		data={'kmeans', centers, radi};
		y=check_accuracy(clustering_accuracy, data);

	elseif strcmp(clustering_name, 'Hierarchical')==1
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
		error('Unsupported clustering %s', clustering_name);
	end
