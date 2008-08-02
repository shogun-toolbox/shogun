check_accuracy <- function(accuracy, type, data) {
	if (regexpr(type, 'kernel')>0) {
		print(paste('train:', data[1], ', test:', data[2], '<--- accuracy', accuracy))
	} else if (regexpr(type, 'classifier')>0) {
		print(paste('alphas:', data[1], ', bias:', data[2], ', sv:', data[3], ', classified:', data[4], '<--- accuracy ', accuracy))
	} else if (regexpr(type, 'distribution')>0) {
		print(paste('likelihood:', data[1], ', derivatives:', data[2], '<--- accuracy ', accuracy))
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
