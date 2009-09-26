def svm_light ():
	print 'SVMLight'

	size_cache=10
	degree=20
	C=0.017
	epsilon=1e-5
	use_bias=False

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)

	sg('set_labels', 'TRAIN', label_train_dna)

	try:
		sg('new_classifier', 'SVMLIGHT')
	except RuntimeError:
		return

	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	result=sg('classify')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')
	svm_light()
