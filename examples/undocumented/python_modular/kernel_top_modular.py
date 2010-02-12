def top():
	print "TOP Kernel"
	from shogun.Features import StringCharFeatures, StringWordFeatures, TOPFeatures, DNA
	from shogun.Kernel import PolyKernel
	from shogun.Distribution import HMM, BW_NORMAL

	N=1 # toy HMM with 1 state 
	M=4 # 4 observations -> DNA
	pseudo=1e-1
	order=1
	gap=0
	reverse=False
	kargs=[1, False, True]

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
	feats_train=TOPFeatures(10, pos, neg, False, False)
	kernel=PolyKernel(feats_train, feats_train, *kargs)
	km_train=kernel.get_kernel_matrix()

	# get kernel on testing data
	pos_clone=HMM(pos)
	neg_clone=HMM(neg)
	pos_clone.set_observations(wordfeats_test)
	neg_clone.set_observations(wordfeats_test)
	feats_test=TOPFeatures(10, pos_clone, neg_clone, False, False)
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import where
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')

	fm_hmm_pos=[ fm_train_dna[i] for i in where([label_train_dna==1])[1] ]
	fm_hmm_neg=[ fm_train_dna[i] for i in where([label_train_dna==-1])[1] ]
	top()
