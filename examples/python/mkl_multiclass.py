def mkl_multiclass ():
	print 'mkl_multiclass'

	size_cache=10
	width=2.1
	C=1.2
	epsilon=1e-5
	mkl_eps=0.01
	mkl_norm=1

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

	sg('set_labels', 'TRAIN', label_train_multiclass)
	sg('new_classifier', 'MKL_MULTICLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('mkl_parameters', mkl_eps, 0, mkl_norm)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_multiclass=lm.load_labels('../data/label_train_multiclass.dat')
	gmnpsvm()
