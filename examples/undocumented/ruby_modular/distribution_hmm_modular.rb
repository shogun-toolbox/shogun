# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
# *** data=LoadMatrix.load_cubes('../data/fm_train_cube.dat')
data=Modshogun::LoadMatrix.new
data.set_features.load_cubes('../data/fm_train_cube.dat')

parameter_list=[[data, 1, 64, 1e-5, 2, 0, False, 5], [data, 3, 6, 1e-1, 1, 0, False, 2]]

def distribution_hmm_modular(fm_cube, N, M, pseudo, order, gap, reverse, num_examples)

# *** 	charfeat=StringCharFeatures(CUBE)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(CUBE)
	charfeat.set_features(fm_cube)
# *** 	feats=StringWordFeatures(charfeat.get_alphabet())
	feats=Modshogun::StringWordFeatures.new
	feats.set_features(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

# *** 	hmm=HMM(feats, N, M, pseudo)
	hmm=Modshogun::HMM.new
	hmm.set_features(feats, N, M, pseudo)
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

	lik_example = hmm.get_log_likelihood()
	lik_sample = hmm.get_log_likelihood_sample()

	return lik_example, lik_sample, hmm


end
###########################################################################
# call functions
###########################################################################

if __FILE__ == $0
	puts 'HMM'
	distribution_hmm_modular(*parameter_list[0])

end
