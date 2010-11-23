from tools.load import LoadMatrix
from numpy import ushort
lm=LoadMatrix()
parameter_list=[[ushort(lm.load_numbers('../data/fm_train_word.dat')),ushort(lm.load_numbers('../data/fm_test_word.dat')),1.2],[ushort(lm.load_numbers('../data/fm_train_word.dat')),ushort(lm.load_numbers('../data/fm_test_word.dat')),1.2]]

def kernel_linear_word_modular (fm_train_word=ushort(lm.load_numbers('../data/fm_train_word.dat')),fm_test_word=ushort(lm.load_numbers('../data/fm_test_word.dat')),scale=1.2):
	print 'LinearWord'
	from shogun.Kernel import LinearKernel, AvgDiagKernelNormalizer
	from shogun.Features import WordFeatures
	fm_train_word=fm_train_word
	fm_test_word = fm_test_word
	scale = scale

	feats_train=WordFeatures(fm_train_word)
	feats_test=WordFeatures(fm_test_word)

	kernel=LinearKernel(feats_train, feats_train)
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	print km_train
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import ushort
	lm=LoadMatrix()
	fm_train_word=ushort(lm.load_numbers('../data/fm_train_word.dat'))
	fm_test_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
	kernel_linear_word_modular(*parameter_list[0])
