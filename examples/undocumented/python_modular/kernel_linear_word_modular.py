def linear_word ():
	print 'LinearWord'
	from shogun.Kernel import LinearKernel, AvgDiagKernelNormalizer
	from shogun.Features import WordFeatures

	feats_train=WordFeatures(fm_train_word)
	feats_test=WordFeatures(fm_test_word)
	scale=1.4

	kernel=LinearKernel(feats_train, feats_train)
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import ushort
	lm=LoadMatrix()
	fm_train_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
	fm_test_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
	linear_word()
