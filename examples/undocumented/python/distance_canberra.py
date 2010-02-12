def canberra_metric ():
	print 'CanberraMetric'
	from sg import sg
	sg('set_distance', 'CANBERRA', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	dm=sg('get_distance_matrix', 'TRAIN')

	sg('set_features', 'TEST', fm_test_real)
	dm=sg('get_distance_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	canberra_metric()
