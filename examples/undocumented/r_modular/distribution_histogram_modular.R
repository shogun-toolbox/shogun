library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))

# Histogram
print('Histogram')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(charfeat, fm_train_dna)
feats=StringWordFeatures(charfeat$get_alphabet())
dump <- feats$obtain_from_char(feats, charfeat, start, order, gap, reverse)
preproc=SortWordString()
dump <- preproc$init(preproc, feats)
dump <- feats$add_preproc(feats, preproc)
dump <- feats$apply_preproc(feats)

histo=Histogram(feats)
dump <- histo$train(histo)

dump <- histo$get_histogram()

num_examples <- feats$get_num_vectors()
num_param <- histo$get_num_model_parameters()

# commented out as this is quite time consuming
#derivs=matrix(0,num_param, num_examples)
#for (i in 0:(num_examples-1))
#{
#	for (j in 0:(num_param-1))
#	{
#		derivs[j,i]=histo$get_log_derivative(histo, j, i)
#	}
#}
dump <- histo$get_log_likelihood(histo)
dump <- histo$get_log_likelihood_sample()
