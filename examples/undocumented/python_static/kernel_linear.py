from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.2,10],[traindat,testdat,1.5,11]]

def kernel_linear (fm_train_real=traindat,fm_test_real=testdat,
		scale=1.2,size_cache=10):

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('Linear')
	kernel_linear(*parameter_list[0])
