from tools.load import LoadMatrix
from numpy import where
lm=LoadMatrix()

parameter_list = [[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),lm.load_labels('../data/label_train_dna.dat'),1,4,1e-1,1,0,False,[1,False,True]],[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),lm.load_labels('../data/label_train_dna.dat'),1,4,1e-1,1,0,False,[1,False,True]]]

def kernel_fisher_modular(fm_train_dna=lm.load_dna('../data/fm_train_dna.dat'),fm_test_dna=lm.load_dna('../data/fm_test_dna.dat'),label_train_dna=lm.load_labels('../data/label_train_dna.dat'), N=1,M=4,pseudo=1e-1,order=1,gap=0,reverse=False,kargs=[1,False,True]):
	print "Fisher Kernel"
	from shogun.Features import StringCharFeatures, StringWordFeatures, FKFeatures, DNA
	from shogun.Kernel import PolyKernel
	from shogun.Distribution import HMM, BW_NORMAL
	
	fm_train_dna = fm_train_dna
	fm_test_dna = fm_test_dna
	label_train_dna = label_train_dna
	fm_hmm_pos=[ fm_train_dna[i] for i in where([label_train_dna==1])[1] ]
	fm_hmm_neg=[ fm_train_dna[i] for i in where([label_train_dna==-1])[1] ]

	N=N # toy HMM with 1 state 
	M=M # 4 observations -> DNA
	pseudo=pseudo
	order=order
	gap=gap
	reverse=reverse
	kargs=kargs

	# train HMM for positive class
	charfeat=StringCharFeatures(fm_hmm_pos, DNA)
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
	print km_train
	# get kernel on testing data
	pos_clone=HMM(pos)
	neg_clone=HMM(neg)
	pos_clone.set_observations(wordfeats_test)
	neg_clone.set_observations(wordfeats_test)
	feats_test=FKFeatures(10, pos_clone, neg_clone)
	feats_test.set_a(feats_train.get_a()) #use prior from training data
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test
if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import where
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')

	fm_hmm_pos=[ fm_train_dna[i] for i in where([label_train_dna==1])[1] ]
	fm_hmm_neg=[ fm_train_dna[i] for i in where([label_train_dna==-1])[1] ]
	kernel_fisher_modular(*parameter_list[0])
