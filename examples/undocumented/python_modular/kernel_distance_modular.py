from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

parameter_list=[[double(lm.load_numbers('../data/fm_train_real.dat')),double(lm.load_numbers('../data/fm_test_real.dat')),1.7],[double(lm.load_numbers('../data/fm_train_real.dat')),double(lm.load_numbers('../data/fm_test_real.dat')),1.8]]

def kernel_distance_modular (fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat')),fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat')),width=1.7):
	print 'Distance'
	from shogun.Kernel import DistanceKernel
	from shogun.Features import RealFeatures
	from shogun.Distance import EuclidianDistance
	fm_train_real = fm_train_real
	fm_test_real  = fm_test_real
	width         =width
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	distance=EuclidianDistance()

	kernel=DistanceKernel(feats_train, feats_test, width, distance)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat'))
	kernel_distance_modular(*parameter_list[0])
