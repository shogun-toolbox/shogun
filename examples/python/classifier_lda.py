def lda ():
	print 'LDA'

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'LDA')
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')

if __name__=='__main__': #svm_light()
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
	lda()
