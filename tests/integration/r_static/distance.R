distance <- function(filename) {
	source('util/set_features.R')
	source('util/set_distance.R')
	source('util/check_accuracy.R')

	if (!set_features('distance_')) {
		return(TRUE)
	}

	if (!set_distance()) {
		return(TRUE)
	}

	dmatrix <- sg('get_distance_matrix', 'TRAIN')
	dm_train <- max(max(abs(distance_matrix_train-dmatrix)))

	dmatrix <- sg('get_distance_matrix', 'TEST')
	dm_test <- max(max(abs(distance_matrix_test-dmatrix)))

	data <- list(dm_train, dm_test)
	return(check_accuracy(distance_accuracy, 'distance', data))
}
