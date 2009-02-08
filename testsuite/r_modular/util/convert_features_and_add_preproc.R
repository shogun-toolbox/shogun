convert_features_and_add_preproc <- function(char_train, char_test, prefix) {
	source('util/tobool.R')

	if (!exists(paste(prefix, 'order', sep=''), mode='numeric')) {
		return(FALSE)
	}

	ftype <- eval(parse(text=paste(prefix, 'feature_type', sep='')))
	if (regexpr('Ulong', ftype)>0) {
		train <- StringUlongFeatures(char_train$get_alphabet())
		test <- StringUlongFeatures(char_test$get_alphabet())
		preproc <- SortUlongString()
	} else if (regexpr('Word', ftype)>0) {
		train <- StringWordFeatures(char_train$get_alphabet())
		test <- StringWordFeatures(char_test$get_alphabet())
		preproc <- SortWordString()
	} else {
		return(FALSE)
	}

	order <- as.integer(eval(parse(text=paste(prefix, 'order', sep=''))))
	start <- as.integer(order-1)
	gap <- as.integer(eval(parse(text=paste(prefix, 'gap', sep=''))))
	reverse <- tobool(eval(parse(text=paste(prefix, 'reverse', sep=''))))
	train$obtain_from_char(train, char_train, start, order, gap, reverse)
	test$obtain_from_char(test, char_test, start, order, gap, reverse)

	preproc$init(preproc, train)
	train$add_preproc(train, preproc)
	train$apply_preproc(train)
	test$add_preproc(test, preproc)
	test$apply_preproc(test)

	return(list(train, test))
}
