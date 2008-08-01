set_features <- function() {
	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')

	sg('set_features', 'TRAIN', data_train)
	sg('set_features', 'TEST', data_test)

	return(TRUE)
}
