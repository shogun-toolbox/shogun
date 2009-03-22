def match_word_string ():
	print 'MatchWordString'
	from shogun.Kernel import MatchWordStringKernel, AvgDiagKernelNormalizer
	from shogun.Features import StringWordFeatures, StringCharFeatures, DNA

	degree=3
	scale=1.4
	size_cache=10
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(DNA)
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(DNA)
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	kernel=MatchWordStringKernel(size_cache, degree)
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	match_word_string()
