library(modshogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))

# Linear HMM
print('LinearHMM')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_train_dna)
feats <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats$obtain_from_char(feats, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats)
dump <- feats$add_preproc(feats, preproc)
dump <- feats$apply_preproc(feats)

hmm <- LinearHMM(feats)
dump <- hmm$train(hmm)

dump <- hmm$get_transition_probs()

num_examples <- feats$get_num_vectors()
num_param <- hmm$get_num_model_parameters()
derivs <- matrix(0, num_param, num_examples)

for (i in 0:(num_examples-1))
{
	for (j in 0:(num_param-1))
	{
		derivs[j,i] <- hmm$get_log_derivative(hmm, j, i)
	}
}

dump <- hmm$get_log_likelihood(hmm)
dump <- hmm$get_log_likelihood_sample()
