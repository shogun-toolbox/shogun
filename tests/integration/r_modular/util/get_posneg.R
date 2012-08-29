get_posneg <- function(prefix) {
	print('Datatype Word not yet supported in R')
	return(FALSE) # data type word not yet supported in R
	N <- as.integer(eval(parse(text=paste(prefix, 'N', sep=''))))
	M <- as.integer(eval(parse(text=paste(prefix, 'M', sep=''))))
	pseudo <- eval(parse(text=paste(prefix, 'pseudo', sep='')))
	order <- as.integer(eval(parse(text=paste(prefix, 'order', sep=''))))
	start <- as.integer(order-1)
	gap <- as.integer(eval(parse(text=paste(prefix, 'gap', sep=''))))
	reverse <- tobool(paste(prefix, 'reverse', sep=''))
	data_train <- eval(parse(text=paste(prefix, 'data_train', sep='')))
	data_test <- eval(parse(text=paste(prefix, 'data_test', sep='')))

	charfeat <- StringCharFeatures(CUBE)
	charfeat$set_string_features(charfeat, data_train)
	wordfeats_train <- StringWordFeatures(charfeat$get_alphabet())
	wordfeats_train$obtain_from_char(
		wordfeats_train, charfeat, start, order, gap, reverse)
	preproc <- SortWordString()
	preproc$init(preproc, wordfeats_train)
	wordfeats_train$add_preproc(wordfeats_train, preproc)
	wordfeats_train$apply_preproc(wordfeats_train)

	charfeat <- StringCharFeatures(CUBE)
	charfeat$set_string_features(charfeat, data_test)
	wordfeats_test <- StringWordFeatures(charfeat$get_alphabet())
	wordfeats_test$obtain_from_char(
		wordfeats_test, charfeat, start, order, gap, reverse)
	wordfeats_test$add_preproc(wordfeats_test, preproc)
	wordfeats_test$apply_preproc(wordfeats_test)

	pos_train <- HMM(wordfeats_train, N, M, pseudo)
	pos_train$train(pos_train)
	pos_train$baum_welch_viterbi_train(pos_train, 'BW_NORMAL')
	neg_train <- HMM(wordfeats_train, N, M, pseudo)
	neg_train$train(neg_train)
	neg_train$baum_welch_viterbi_train(neg_train, 'BW_NORMAL')
	pos_test=HMM(pos_train)
	pos_test$set_observations(pos_test, wordfeats_test)
	neg_test=HMM(neg_train)
	neg_test$set_observations(neg_test, wordfeats_test)

	return(list(list(pos_train, neg_train), list(pos_test, neg_test)))
}

