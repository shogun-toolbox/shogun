distance <- function(filename) {
	source('util/set_features.R')
	source('util/set_distance.R')
	source('util/check_accuracy.R')

	if (!set_features()) {
		return(TRUE)
	}

	if (!set_distance()) {
		return(TRUE)
	}

	dmatrix <- sg('get_distance_matrix')
	dtrain <- max(max(abs(dm_train-dmatrix)))

	sg('init_distance', 'TEST')
	dmatrix <- sg('get_distance_matrix')
	dtest <- max(max(abs(dm_test-dmatrix)))

	data <- list(dtrain, dtest)
	return(check_accuracy(accuracy, 'distance', data))
}
