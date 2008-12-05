distance <- function(filename) {
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

	dmatrix <- distance$get_distance_matrix()
	dm_train <- max(max(abs(distance_matrix_train-dmatrix)))

	distance$init(distance, feats[[1]], feats[[2]])
	dmatrix <- distance$get_distance_matrix()
	dm_test <- max(max(abs(distance_matrix_test-dmatrix)))

	data <- list(dm_train, dm_test)
	return(check_accuracy(distance_accuracy, 'distance', data))
}
