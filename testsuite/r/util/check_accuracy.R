check_accuracy <- function(accuracy, type, data) {
	if (regexpr(type, 'kernel')) {
		print(paste('train:', data[1], ', test ', data[2], ' <--- accuracy ', accuracy))
	} else {
		print(paste('Unknown accuracy type', type, '!'))
		return(FALSE)
	}

	for (i in 1:length(data)) {
		if (data[i]>accuracy) {
			return(FALSE)
		}
	}

	return(TRUE)
}
