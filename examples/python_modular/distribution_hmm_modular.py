def hmm ():
	print 'HMM'

	from shogun.Features import StringWordFeatures, StringCharFeatures, CUBE
	from shogun.Distribution import HMM, BW_NORMAL

	N=3
	M=6
	pseudo=1e-1
	order=1
	gap=0
	reverse=False
	num_examples=2

	charfeat=StringCharFeatures(CUBE)
	charfeat.set_features(fm_cube)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

	hmm=HMM(feats, N, M, pseudo)
	hmm.train()
	hmm.baum_welch_viterbi_train(BW_NORMAL)

	num_examples=feats.get_num_vectors()
	num_param=hmm.get_num_model_parameters()
	for i in xrange(num_examples):
		for j in xrange(num_param):
			hmm.get_log_derivative(j, i)

	best_path=0
	best_path_state=0
	for i in xrange(num_examples):
		best_path+=hmm.best_path(i)
		for j in xrange(N):
			best_path_state+=hmm.get_best_path_state(i, j)

	hmm.get_log_likelihood()
	hmm.get_log_likelihood_sample()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_cube=lm.load_cubes('../data/fm_train_cube.dat')
	hmm()
