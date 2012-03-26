from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.9],[traindat,testdat,1.7]]

def kernel_io_modular (fm_train_real=traindat,fm_test_real=testdat,width=1.9):
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.IO import AsciiFile, BinaryFile
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)


	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()
	f=AsciiFile("gaussian_train.ascii","w")
	kernel.save(f)
	del f

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	f=AsciiFile("gaussian_test.ascii","w")
	kernel.save(f)
	del f

	#clean up
	import os
	os.unlink("gaussian_test.ascii")
	os.unlink("gaussian_train.ascii")
	
	return km_train, km_test, kernel

if __name__=='__main__':
	print('Gaussian')
	kernel_io_modular(*parameter_list[0])
