from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.,10],[traindat,testdat,1.5,11]]

def kernel_combined(fm_train_real=traindat,fm_test_real=testdat,
		 weight=1.,size_cache=10):
	sg('clean_kernel')
	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')
	sg('set_kernel', 'COMBINED', size_cache)
	sg('add_kernel', weight, 'LINEAR', 'REAL', size_cache)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', size_cache, 1.)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'POLY', 'REAL', size_cache, 3, False)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)

	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('Combined')
	kernel_combined(*parameter_list[0])
