def combined ():
	print 'Combined'

	size_cache=10
	weight=1.

	from sg import sg
	sg('clean_kernel')
	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')
	sg('set_kernel', 'COMBINED', size_cache)
	sg('add_kernel', weight, 'LINEAR', 'REAL', size_cache)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', size_cache, 1.)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'POLY', 'REAL', size_cache, 3, False)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)

	km=sg('get_kernel_matrix, 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	combined()
