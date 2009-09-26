def distance ():
	print 'Distance'

	width=1.7
	size_cache=10

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('set_kernel', 'DISTANCE', size_cache, width)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST")

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	distance()
