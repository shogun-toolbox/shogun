def gpbtsvm ():
	print 'GPBTSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'GPBTSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
	gpbtsvm()
