from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
trainlabel=lm.load_labels('../data/label_train_multiclass.dat')
parameter_list=[[traindat,testdat,trainlabel,10,1.2,1.2,1e-5,0.001,1.5,1.0],
		[traindat,testdat,trainlabel,11,1.3,1.3,1e-5,0.002,1.6,1.1]]

def mkl_multiclass (fm_train_real=traindat,fm_test_real=testdat,
		label_train_multiclass=trainlabel,
		size_cache=10,width=1.2,C=1.2,epsilon=1e-5,
		mkl_eps=0.001,mkl_norm=1.5,weight=1.0):

	sg('clean_kernel')
	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')
	sg('set_kernel', 'COMBINED', size_cache)
	sg('add_kernel', weight, 'LINEAR', 'REAL', size_cache)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', size_cache, width)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'POLY', 'REAL', size_cache, 2)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)

	sg('set_labels', 'TRAIN', label_train_multiclass)
	sg('new_classifier', 'MKL_MULTICLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('mkl_parameters', mkl_eps, 0.0, mkl_norm)
	sg('train_classifier')

	#sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')
	return result

if __name__=='__main__':
	print('mkl_multiclass')
	mkl_multiclass(*parameter_list[0])
