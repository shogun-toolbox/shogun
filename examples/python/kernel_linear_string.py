def linear_string ():
	print 'LinearString'

	size_cache=10

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LINEAR', 'CHAR', size_cache)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	linear_string()
