# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 2.0],[traindat,testdat, 3.0]]

def kernel_tstudent_modular(fm_train_real=traindat,fm_test_real=testdat, degree=2.0)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)
	
# *** 	distance=EuclidianDistance(feats_train, feats_train)
	distance=Modshogun::EuclidianDistance.new
	distance.set_features(feats_train, feats_train)

# *** 	kernel=TStudentKernel(feats_train, feats_train, degree, distance)
	kernel=Modshogun::TStudentKernel.new
	kernel.set_features(feats_train, feats_train, degree, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel



end
if __FILE__ == $0
	puts 'TStudent'
	kernel_tstudent_modular(*parameter_list[0])

end
