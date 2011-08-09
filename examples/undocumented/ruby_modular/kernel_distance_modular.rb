# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = double(LoadMatrix.load_numbers('../data/fm_test_real.dat'))
testdat = double(LoadMatrix.load_numbers('../data/fm_train_real.dat'))
parameter_list=[[traindat,testdat,1.7],[traindat,testdat,1.8]]

def kernel_distance_modular(fm_train_real=traindat,fm_test_real=testdat,width=1.7)
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	distance=EuclidianDistance()

	kernel=DistanceKernel(feats_train, feats_test, width, distance)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'Distance'
	kernel_distance_modular(*parameter_list[0])

end
