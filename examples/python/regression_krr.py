def krr ():
	print 'KRR'

	size_cache=10
	width=2.1
	C=1.2
	tau=1e-6

	from sg import sg
	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train)

	sg('new_regression', 'KRR')
	sg('krr_tau', tau)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	result=sg('classify')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	fm_test=lm.load_numbers('../data/fm_test_real.dat')
	label_train=lm.load_labels('../data/label_train_twoclass.dat')
	krr()
