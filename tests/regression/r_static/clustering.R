clustering <- function(filename) {
	source('util/set_features.R')
	source('util/set_distance.R')
	source('util/check_accuracy.R')
	source('util/fix_clustering_name_inconsistency.R')

	if (!set_features('distance_')) {
		return(TRUE)
	}

	if (!set_distance()) {
		return(TRUE)
	}

	cname <- fix_clustering_name_inconsistency(clustering_name)
	sg('new_clustering', cname)

	if (exists('clustering_max_iter')) {
		max_iter <- clustering_max_iter
	} else {
		max_iter <- 1000
	}

	if (exists('clustering_k')) {
		first_arg <- clustering_k
	} else if (exists('clustering_merges')) {
		first_arg <- clustering_merges
	} else {
		print('Incomplete clustering data!')
		return(FALSE)
	}

	sg('train_clustering', first_arg, max_iter)

	if (exists('clustering_radi')) {
		res <- sg('get_clustering')
		radi <- t(res[[1]])
		radi <- max(abs(radi-clustering_radi))
		centers <- max(max(abs(res[[2]]-clustering_centers)))

		data <- list(centers, radi);
		return(check_accuracy(clustering_accuracy, 'kmeans', data))

	} else if (exists('clustering_merge_distance')) {
		res <- sg('get_clustering')
		merge_distances <- t(res[[1]])
		merge_distances <- max(abs(
			merge_distances-clustering_merge_distance))
		pairs <- max(max(abs(res[[2]]-clustering_pairs)))

		data <- list(merge_distances, pairs)
		return(check_accuracy(clustering_accuracy, 'hierarchical', data))

	} else {
		print('Incomplete clustering data!')
		return(FALSE)
	}
}
