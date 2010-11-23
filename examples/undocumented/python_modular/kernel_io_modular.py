from tools.load import LoadMatrix
lm=LoadMatrix()

parameter_list=[[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.9],[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.7]]

def kernel_io_modular (fm_train_real=lm.load_numbers('../data/fm_train_real.dat'),fm_test_real=lm.load_numbers('../data/fm_test_real.dat'),width=1.9):
	print 'Gaussian'
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Library import AsciiFile, BinaryFile
	fm_train_real=fm_train_real
	fm_test_real=fm_test_real
	width = width
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)


	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()
	f=AsciiFile("gaussian_train.ascii","w")
	kernel.save(f)
	del f

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test
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
	kernel_io_modular(*parameter_list[0])
