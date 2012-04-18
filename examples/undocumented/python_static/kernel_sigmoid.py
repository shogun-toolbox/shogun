from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,11,1.2,1.3,10],[traindat,testdat,12,1.3,1.4,11]]

def kernel_sigmoid (fm_train_real=traindat,fm_test_real=testdat,
		 num_feats=11,gamma=1.2,coef0=1.3,size_cache=10):

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('Sigmoid')
	kernel_sigmoid(*parameter_list[0])
