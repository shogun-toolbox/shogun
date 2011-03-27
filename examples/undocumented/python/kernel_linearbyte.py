from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

trainbyte=ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
testbyte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))

parameter_list=[[trainbyte,testbyte],[trainbyte,testbyte]]

def kernel_linearbyte (fm_train_byte=trainbyte,fm_test_byte=testbyte):

	#import pdb
	#pdb.set_trace()
	sg('set_features', 'TRAIN', fm_train_byte)
	sg('set_features', 'TEST', fm_test_byte, 'RAWBYTE')
	sg('set_kernel', 'LINEAR', 'BYTE', 10)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print 'LinearByte'
	kernel_linearbyte(*parameter_list[0])
