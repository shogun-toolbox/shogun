check_accuracy <- function(accuracy, type, data) {
	print(data)
	print(paste('train: ', data[1], 'test ', data[2], ' <--- accuracy ', accuracy))

	for (i in 1:length(data)) {
		if (data[i]>accuracy) {
			return(FALSE);
		}
	}

	return(TRUE);
}
