from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
trainlabel=lm.load_labels('../data/label_train_regression.dat')
parameter_list=[[traindat,testdat,trainlabel,10,2.1,1.2,1e-5,1e-2],
		[traindat,testdat,trainlabel,11,2.3,1.3,1e-6,1e-3]]

def regression_libsvr (fm_train=traindat,fm_test=testdat,
		label_train=trainlabel,size_cache=10,width=2.1,
		C=1.2,epsilon=1e-5,tube_epsilon=1e-2):

	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train)
	sg('new_regression', 'LIBSVR')
	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	result=sg('classify')
	return result

if __name__=='__main__':
	print('LibSVR')
	regression_libsvr(*parameter_list[0])
