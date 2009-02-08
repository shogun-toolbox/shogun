clustering <- function(filename) {
	source('util/get_features.R')
	source('util/get_distance.R')
	source('util/check_accuracy.R')

	feats <- get_features('distance_')
	if (typeof(feats)=='logical') {
		return(TRUE)
	}

	distance <- get_distance(feats)
	if (typeof(distance)=='logical') {
		return(TRUE)
	}

	if (regexpr('KMeans', clustering_name)>0) {
		clustering <- KMeans(as.integer(clustering_k), distance);
		clustering$train()

		distance$init(distance, feats[[1]], feats[[2]])
		centers <- clustering$get_cluster_centers()
		centers <- max(max(abs(centers-clustering_centers)))
		radi <- clustering$get_radiuses()
		radi <- max(abs(radi-clustering_radi))
		data <- list(centers, radi)

		return(check_accuracy(clustering_accuracy, 'kmeans', data))
	} else if (regexpr('Hierarchical', clustering_name)>0) {
		clustering <- Hierarchical(as.integer(clustering_merges), distance);
		clustering$train()

		distance$init(distance, feats[[1]], feats[[2]])
		merge_distances <- clustering$get_merge_distances()
		merge_distances <- max(abs(merge_distances-clustering_merge_distance))
		pairs <- clustering$get_cluster_pairs()
		pairs <- max(max(abs(pairs-clustering_pairs)))
		data <- list(merge_distances, pairs)

		return(check_accuracy(clustering_accuracy, 'hierarchical', data))
	} else {
		print('Incomplete clustering data!')
		return(FALSE)
	}
}
