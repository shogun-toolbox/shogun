def gaussian ():
	print 'Gaussian'

	width=1.9
	size_cache=10

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	gaussian()
