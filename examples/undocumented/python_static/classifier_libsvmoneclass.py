from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()


traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,10,2.1,10.,1e-5,False],
		[traindat,testdat,10,2.1,11.,1e-4,False]]

def classifier_libsvm_oneclass (fm_train_real=traindat,fm_test_real=testdat,
			size_cache=10, width=2.1,C=10.,
			epsilon=1e-5,use_bias=False):

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('new_classifier', 'LIBSVM_ONECLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')
	kernel_matrix = sg('get_kernel_matrix', 'TEST')
	return result, kernel_matrix

if __name__=='__main__':
	print('LibSVMOneClass')
	classifier_libsvm_oneclass(*parameter_list[0])
