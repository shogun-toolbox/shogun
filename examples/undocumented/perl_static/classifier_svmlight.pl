from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()


traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
train_label=lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindna,testdna, train_label,10,20,1.2,1e-5,False],
		[traindna,testdna,train_label,10,21,1.3,1e-4,False]]

def classifier_svmlight (fm_train_dna=traindna,fm_test_dna=testdna,label_train_dna=train_label,
			size_cache=10, degree=20,C=1.2,
			epsilon=1e-5,use_bias=False):


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
	kernel_matrix = sg('get_kernel_matrix', 'TEST')
	return result, kernel_matrix

if __name__=='__main__':
	print('SVMLight')
	classifier_svmlight(*parameter_list[0])
