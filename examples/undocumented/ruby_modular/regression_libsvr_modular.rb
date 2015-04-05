require 'rubygems'
require 'modshogun'
require 'pp'
require 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5,1e-2], [traindat,testdat,label_traindat,2.1,1,1e-5,1e-2]]


def regression_libsvr_modular(fm_train=traindat,fm_test=testdat,label_train=label_traindat,width=2.1,c=1,epsilon=1e-5,tube_epsilon=1e-2)


	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test)

	kernel=Modshogun::GaussianKernel.new(feats_train, feats_train, width)
	labels=Modshogun::RegressionLabels.new(label_train)

	svr=Modshogun::LibSVR.new(c, tube_epsilon, kernel, labels)
	svr.set_epsilon(epsilon)
	svr.train()

	kernel.init(feats_train, feats_test)
	out1=svr.apply().get_labels()
	out2=svr.apply(feats_test).get_labels()

	return out1,out2,kernel

end

if __FILE__ == $0
	puts 'LibSVR'
	pp regression_libsvr_modular(*parameter_list[0])
end
