from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

traindat = double(lm.load_numbers('../data/fm_test_real.dat'))
testdat = double(lm.load_numbers('../data/fm_train_real.dat'))
parameter_list=[[traindat,testdat,1.7],[traindat,testdat,1.8]]

def kernel_distance_modular (fm_train_real=traindat,fm_test_real=testdat,width=1.7):
	from shogun.Kernel import DistanceKernel
	from shogun.Features import RealFeatures
	from shogun.Distance import EuclidianDistance
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	distance=EuclidianDistance()

	kernel=DistanceKernel(feats_train, feats_test, width, distance)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Distance')
	kernel_distance_modular(*parameter_list[0])
