def linear_hmm ():
	print 'LinearHMM'

	from shogun.Features import StringWordFeatures, StringCharFeatures, DNA
	from shogun.Distribution import LinearHMM

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_dna)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

	hmm=LinearHMM(feats)
	hmm.train()

	hmm.get_transition_probs()

	num_examples=feats.get_num_vectors()
	num_param=hmm.get_num_model_parameters()
	for i in xrange(num_examples):
		for j in xrange(num_param):
			hmm.get_log_derivative(j, i)

	hmm.get_log_likelihood()
	hmm.get_log_likelihood_sample()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_dna=lm.load_dna('../data/fm_train_dna.dat')
	linear_hmm()
