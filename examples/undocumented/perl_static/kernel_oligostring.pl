from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,10,3,1.2],
		[traindna,testdna,11,4,1.3]]

def kernel_oligostring (fm_train_dna=traindna,fm_test_dna=testdna,
			    size_cache=10,k=3,width=1.2):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'OLIGO', 'CHAR', size_cache, k, width)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('OligoString')
	kernel_oligostring(*parameter_list[0])
