#!/usr/bin/env python
import shogun as sg
from tools.load import LoadMatrix
lm=LoadMatrix()
data=lm.load_cubes('../data/fm_train_cube.dat')

parameter_list=[[data, 1, 64, 1e-5, 2, 0, False, 5], [data, 3, 6, 1e-1, 1, 0, False, 2]]

def distribution_hmm(fm_cube, N, M, pseudo, order, gap, reverse, num_examples):

	charfeat=sg.create_string_features(fm_cube, sg.CUBE)
	feats=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))

	# FIXME: add HMM to new parameter framework
	# hmm=sg.HMM(feats, N, M, pseudo)
	# hmm.train()
	# hmm.baum_welch_viterbi_train(sg.BW_NORMAL)

	# num_examples=feats.get_num_vectors()
	# num_param=hmm.get_num_model_parameters()
	# for i in range(num_examples):
	# 	for j in range(num_param):
	# 		hmm.get_log_derivative(j, i)

	# best_path=0
	# best_path_state=0
	# for i in range(num_examples):
	# 	best_path+=hmm.best_path(i)
	# 	for j in range(N):
	# 		best_path_state+=hmm.get_best_path_state(i, j)

	# lik_example = hmm.get_log_likelihood()
	# lik_sample = hmm.get_log_likelihood_sample()

	# return lik_example, lik_sample, hmm

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	print('HMM')
	distribution_hmm(*parameter_list[0])
