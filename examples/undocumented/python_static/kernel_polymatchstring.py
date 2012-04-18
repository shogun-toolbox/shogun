from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,10,3,False],
		[traindna,testdna,11,4,False]]

def kernel_polymatchstring (fm_train_dna=traindna,fm_test_dna=testdna,
			    size_cache=10,degree=3,inhomogene=False):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('PolyMatchString')
	kernel_polymatchstring(*parameter_list[0])
