function y = clustering(filename)
	addpath('util');
	addpath('../data/clustering');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	% there is some randomness involved, alas this is
	% not working correctly in matlab
	rand('state', init_random);

	if ~set_features('distance_')
		return;
	end

	if ~set_distance()
		return;
	end

	cname=fix_clustering_name_inconsistency(clustering_name);
	sg('new_clustering', cname);

	if ~isempty(clustering_max_iter)
		max_iter=clustering_max_iter;
	else
		max_iter=1000;
	end

	if ~isempty(clustering_k)
		first_arg=clustering_k;
	elseif ~isempty(clustering_merges)
		first_arg=clustering_merges;
	else
		error('Incomplete clustering data!\n');
	end

	sg('init_random', init_random);
	sg('train_clustering', first_arg, max_iter);

	if ~isempty(clustering_radi)
		[radi, centers]=sg('get_clustering');
		radi=max(abs(radi'-clustering_radi));
		centers=max(max(abs(centers-clustering_centers)));

		data={'kmeans', centers, radi};
		y=check_accuracy(clustering_accuracy, data);

	elseif ~isempty(clustering_merge_distance)
		[merge_distances, pairs]=sg('get_clustering');
		merge_distances=max(abs(merge_distances'-clustering_merge_distance));
		pairs=max(max(abs(pairs-clustering_pairs)));

		data={'hierarchical', merge_distances, pairs};
		y=check_accuracy(clustering_accuracy, data);

	else
		error('Incomplete clustering data!\n');
	end
