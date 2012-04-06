from tools.load import LoadMatrix
from numpy import where

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.3],[traindat,testdat, 1.4]]

def kernel_gaussian_modular (fm_train_real=traindat,fm_test_real=testdat, width=1.3):
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Gaussian')
	kernel_gaussian_modular(*parameter_list[0])
