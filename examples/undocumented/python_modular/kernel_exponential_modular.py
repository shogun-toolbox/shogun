from tools.load import LoadMatrix
from numpy import where

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_exponential_modular (fm_train_real=traindat,fm_test_real=testdat, tau_coef=1.0):
	from shogun.Features import RealFeatures
	from shogun.Kernel import ExponentialKernel
	from shogun.Distance import EuclidianDistance

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance = EuclidianDistance(feats_train, feats_train)
	
	kernel=ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Exponential')
	kernel_exponential_modular(*parameter_list[0])
