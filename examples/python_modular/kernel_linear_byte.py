###########################################################################
# linear kernel on byte features
###########################################################################
def linear_byte():
	print 'LinearByte'
	from shogun.Kernel import LinearByteKernel
	from shogun.Features import ByteFeatures
	
	feats_train=ByteFeatures(fm_train_byte)
	feats_test=ByteFeatures(fm_test_byte)

	kernel=LinearByteKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import ubyte
	lm=LoadMatrix()
	fm_train_byte=ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
	fm_test_byte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))
	linear_byte()
