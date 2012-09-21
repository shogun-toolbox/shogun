from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,3,10],[traindna,testdna,4,11]]

def kernel_fixeddegreestring (fm_train_dna=traindna,fm_test_dna=testdna,degree=3,
			    size_cache=10):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('FixedDegreeString')
	kernel_fixeddegreestring(*parameter_list[0])
