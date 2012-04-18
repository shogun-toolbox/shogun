from tools.load import LoadMatrix
from numpy import ushort
from sg import sg
lm=LoadMatrix()

trainword=ushort(lm.load_numbers('../data/fm_test_word.dat'))
testword=ushort(lm.load_numbers('../data/fm_test_word.dat'))
parameter_list=[[trainword,testword,10,1.4],
	       [trainword,testword,11,1.5]]

def kernel_linearword (fm_train_word=trainword,fm_test_word=testword,
		       size_cache=10, scale=1.4):
	sg('set_features', 'TRAIN', fm_train_word)
	sg('set_features', 'TEST', fm_test_word)
	sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('LinearWord')
	kernel_linearword(*parameter_list[0])
