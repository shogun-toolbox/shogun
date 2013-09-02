#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3,0,False],[traindna,4,0,False]]

def distribution_linearhmm_modular (fm_dna=traindna,order=3,gap=0,reverse=False):

	from modshogun import StringWordFeatures, StringCharFeatures, DNA
	from modshogun import LinearHMM

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_dna)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

	hmm=LinearHMM(feats)
	hmm.train()

	hmm.get_transition_probs()

	num_examples=feats.get_num_vectors()
	num_param=hmm.get_num_model_parameters()
	for i in range(num_examples):
		for j in range(num_param):
			hmm.get_log_derivative(j, i)

	out_likelihood = hmm.get_log_likelihood()
	out_sample = hmm.get_log_likelihood_sample()

	return hmm,out_likelihood ,out_sample
###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	distribution_linearhmm_modular(*parameter_list[0])
	print('LinearHMM')
