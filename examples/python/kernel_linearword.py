def linear_word ():
	print 'LinearWord'

	size_cache=10
	scale=1.4

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_word)
	sg('set_features', 'TEST', fm_test_word)
	sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import ushort
	lm=LoadMatrix()
	fm_train_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
	fm_test_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
	linear_word()
