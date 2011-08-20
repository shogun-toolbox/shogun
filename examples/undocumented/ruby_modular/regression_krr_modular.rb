# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
###########################################################################
# kernel ridge regression
###########################################################################

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,0.8,1e-6],[traindat,testdat,label_traindat,0.9,1e-7]]

def regression_krr_modular(fm_train=traindat,fm_test=testdat,label_train=label_traindat,width=0.8,tau=1e-6)


# *** 	feats_train=RealFeatures(fm_train)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train)
# *** 	feats_test=RealFeatures(fm_test)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test)

# *** 	kernel=GaussianKernel(feats_train, feats_train, width)
	kernel=Modshogun::GaussianKernel.new
	kernel.set_features(feats_train, feats_train, width)

# *** 	labels=Labels(label_train)
	labels=Modshogun::Labels.new
	labels.set_features(label_train)

# *** 	krr=KRR(tau, kernel, labels)
	krr=Modshogun::KRR.new
	krr.set_features(tau, kernel, labels)
	krr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = krr.apply().get_labels()
	return out,kernel,krr


end
# equivialent shorter version
def krr_short()
	puts 'KRR_short'

	width=0.8; tau=1e-6
# *** 	krr=KRR(tau, GaussianKernel(0, width), Labels(label_train))
	krr=Modshogun::KRR.new
	krr.set_features(tau, GaussianKernel(0, width), Labels(label_train))
	krr.train(RealFeatures(fm_train))
	out = krr.apply(RealFeatures(fm_test)).get_labels()

	return krr,out


end
if __FILE__ == $0
	puts 'KRR'
	regression_krr_modular(*parameter_list[0])

end
