require 'modshogun'
require 'pp'
require 'load'
###########################################################################
# kernel ridge regression
###########################################################################

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,0.8,1e-6],[traindat,testdat,label_traindat,0.9,1e-7]]

def regression_krr_modular(fm_train=traindat,fm_test=testdat,label_train=label_traindat,width=0.8,tau=1e-6)


	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test)

	kernel=Modshogun::GaussianKernel.new(feats_train, feats_train, width)

	labels=Modshogun::RegressionLabels.new(label_train)

	krr=Modshogun::KernelRidgeRegression.new(tau, kernel, labels)
	krr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = krr.apply().get_labels()
	return out,kernel,krr

end

# equivialent shorter version
## probably dosn't work yet
def krr_short()
	puts 'KRR_short'

	width=0.8; tau=1e-6
# ***	krr=KernelRidgeRegression(tau, GaussianKernel(0, width), RegressionLabels(label_train))
	krr=Modshogun::KernelRidgeRegression.new(tau, GaussianKernel(0, width), RegressionLabels(label_train))
	#krr.set_features(tau, GaussianKernel(0, width), RegressionLabels(label_train))
	krr.train(RealFeatures(fm_train))
	out = Modshogun::LabelsFactory.to_regression(krr.apply(RealFeatures(fm_test)).get_labels())

	return krr,out

end

if __FILE__ == $0
	puts 'KernelRidgeRegression'
	pp regression_krr_modular(*parameter_list[0])
end
