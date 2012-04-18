from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.4,10],[traindat,testdat,1.5,11]]

def preproc_logplusone (fm_train_real=traindat,fm_test_real=testdat,
		 width=1.4,size_cache=10):

	sg('add_preproc', 'LOGPLUSONE')
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	km=sg('get_kernel_matrix', 'TRAIN')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('LogPlusOne')
	preproc_logplusone(*parameter_list[0])
