function y = clustering(filename)
	y=true;
	addpath('util');
	addpath('../data/clustering');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end
	if !set_distance()
		return;
	end

	cname=fix_clustering_name_inconsistency(name);
	sg('new_clustering', cname)

	if !isempty(clustering_max_iter)
		max_iter=clustering_max_iter;
	else
		max_iter=1000;
	end

	if !isempty(clustering_k)
		first_arg=clustering_k;
	elseif !isempty(clustering_merges)
		first_arg=clustering_merges;
	else
		error('Incomplete clustering data!');
	end

	sg('train_clustering', first_arg, max_iter)

	if !isempty(clustering_radi)
		[radi, centers]=sg('get_clustering');
		radi=max(abs(radi'-clustering_radi));
		centers=max(abs(centers-clustering_centers))(1:1);

		y=check_accuracy_kmeans(clustering_accuracy, radi, centers);

	elseif !isempty(clustering_merge_distance)
		[merge_distances, pairs]=sg('get_clustering');
		merge_distances=max(abs(merge_distances'-clustering_merge_distance));
		pairs=max(abs(pairs-clustering_pairs))(1:1);

		y=check_accuracy_hierarchical(clustering_accuracy, merge_distances, pairs);

	else
		error('Incomplete clustering data!');
	end

