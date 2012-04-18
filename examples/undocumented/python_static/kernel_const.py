from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,23.,10],[traindat,testdat,24.,11]]

def kernel_const (fm_train_real=traindat,fm_test_real=testdat,c=23.,size_cache=10):
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'CONST', 'REAL', size_cache, c)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('Const')
	kernel_const(*parameter_list[0])
