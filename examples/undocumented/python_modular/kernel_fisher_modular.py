#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import where
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')
parameter_list = [[traindat,testdat,label_traindat,1,4,1e-1,1,0,False,[1,False,True]],[traindat,testdat,label_traindat,3,4,1e-1,1,0,False,[1,False,True]]]

fm_hmm_pos=[ traindat[i] for i in where([label_traindat==1])[1] ]
fm_hmm_neg=[ traindat[i] for i in where([label_traindat==-1])[1] ]

def kernel_fisher_modular (fm_train_dna=traindat, fm_test_dna=testdat,
		label_train_dna=label_traindat, 
		N=1,M=4,pseudo=1e-1,order=1,gap=0,reverse=False,
		kargs=[1,False,True]):

	from modshogun import StringCharFeatures, StringWordFeatures, FKFeatures, DNA
	from modshogun import PolyKernel
	from modshogun import HMM, BW_NORMAL#, MSG_DEBUG
	
	# train HMM for positive class
	charfeat=StringCharFeatures(fm_hmm_pos, DNA)
	#charfeat.io.set_loglevel(MSG_DEBUG)
	hmm_pos_train=StringWordFeatures(charfeat.get_alphabet())
	hmm_pos_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	pos=HMM(hmm_pos_train, N, M, pseudo)
	pos.baum_welch_viterbi_train(BW_NORMAL)

	# train HMM for negative class
	charfeat=StringCharFeatures(fm_hmm_neg, DNA)
	hmm_neg_train=StringWordFeatures(charfeat.get_alphabet())
	hmm_neg_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	neg=HMM(hmm_neg_train, N, M, pseudo)
	neg.baum_welch_viterbi_train(BW_NORMAL)

	# Kernel training data
	charfeat=StringCharFeatures(fm_train_dna, DNA)
	wordfeats_train=StringWordFeatures(charfeat.get_alphabet())
	wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	# Kernel testing data
	charfeat=StringCharFeatures(fm_test_dna, DNA)
	wordfeats_test=StringWordFeatures(charfeat.get_alphabet())
	wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	# get kernel on training data
	pos.set_observations(wordfeats_train)
	neg.set_observations(wordfeats_train)
	feats_train=FKFeatures(10, pos, neg)
	feats_train.set_opt_a(-1) #estimate prior
	kernel=PolyKernel(feats_train, feats_train, *kargs)
	km_train=kernel.get_kernel_matrix()

	# get kernel on testing data
	pos_clone=HMM(pos)
	neg_clone=HMM(neg)
	pos_clone.set_observations(wordfeats_test)
	neg_clone.set_observations(wordfeats_test)
	feats_test=FKFeatures(10, pos_clone, neg_clone)
	feats_test.set_a(feats_train.get_a()) #use prior from training data
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print("Fisher Kernel")
	kernel_fisher_modular(*parameter_list[0])
