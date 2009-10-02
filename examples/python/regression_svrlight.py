def svr_light ():
	print 'SVRLight'

	size_cache=10
	width=2.1
	C=1.2
	epsilon=1e-5
	tube_epsilon=1e-2

	from sg import sg
	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train)

	try:
		sg('new_regression', 'SVRLIGHT')
	except RuntimeError:
		return

	sg('svr_tube_epsilon', tube_epsilon)
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
	svr_light()
