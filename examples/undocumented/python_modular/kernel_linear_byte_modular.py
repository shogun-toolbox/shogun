###########################################################################
# linear kernel on byte features
###########################################################################
from tools.load import LoadMatrix
from numpy import ubyte
lm=LoadMatrix()
parameter_list=[[ubyte(lm.load_numbers('../data/fm_train_byte.dat')),ubyte(lm.load_numbers('../data/fm_test_byte.dat'))],[ubyte(lm.load_numbers('../data/fm_train_byte.dat')),ubyte(lm.load_numbers('../data/fm_test_byte.dat'))]]


def kernel_linear_byte_modular(fm_train_byte=ubyte(lm.load_numbers('../data/fm_train_byte.dat')),fm_test_byte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))):
	print 'LinearByte'
	from shogun.Kernel import LinearKernel
	from shogun.Features import ByteFeatures
	fm_train_byte   =fm_train_byte
	fm_test_byte    =fm_test_byte
	feats_train=ByteFeatures(fm_train_byte)
	feats_test=ByteFeatures(fm_test_byte)

	kernel=LinearKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	print km_train
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test
if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import ubyte
	lm=LoadMatrix()
	fm_train_byte=ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
	fm_test_byte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))
	kernel_linear_byte_modular(*parameter_list[0])
