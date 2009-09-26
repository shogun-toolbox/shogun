def chi_square_distance ():
	print 'ChiSquareDistance'
	from sg import sg
	sg('set_distance', 'CHISQUARE', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	dm=sg('get_distance_matrix')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	chi_square_distance()
