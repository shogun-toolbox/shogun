#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3,0,False],[traindna,4,0,False]]

def distribution_histogram_modular (fm_dna=traindna,order=3,gap=0,reverse=False):
	from modshogun import StringWordFeatures, StringCharFeatures, DNA
	from modshogun import Histogram

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_dna)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

	histo=Histogram(feats)
	histo.train()

	histo.get_histogram()

	num_examples=feats.get_num_vectors()
	num_param=histo.get_num_model_parameters()
	#for i in xrange(num_examples):
	#	for j in xrange(num_param):
	#		histo.get_log_derivative(j, i)

	out_likelihood = histo.get_log_likelihood()
	out_sample = histo.get_log_likelihood_sample()
	return histo,out_sample,out_likelihood
###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	print('Histogram')
	distribution_histogram_modular(*parameter_list[0])

