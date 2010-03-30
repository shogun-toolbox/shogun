def gaussian ():
	print 'Gaussian'
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Library import AsciiFile, BinaryFile

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=1.9

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

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	gaussian()
