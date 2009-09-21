def linear_byte ():
	print 'LinearByte'

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_byte, 'RAWBYTE')
	sg('set_features', 'TEST', fm_test_byte, 'RAWBYTE')
	sg('set_kernel', 'LINEAR BYTE', 10)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

if __name__=='__main__':
	from numpy import ubyte
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_byte=ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
	fm_test_byte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))
	linear_byte()
