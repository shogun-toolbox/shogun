from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,4,False,True,10],
		[traindat,testdat,5,False,True,11]]

def kernel_poly (fm_train_real=traindat,fm_test_real=testdat,
		 degree=4,inhomogene=False,use_normalization=True,size_cache=10):

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('Poly')
	kernel_poly(*parameter_list[0])
