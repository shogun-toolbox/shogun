# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_libsvm_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,width=2.1,C=1,epsilon=1e-5)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)
	
# *** 	kernel=GaussianKernel(feats_train, feats_train, width)
	kernel=Modshogun::GaussianKernel.new
	kernel.set_features(feats_train, feats_train, width)
# *** 	labels=Labels(label_train_twoclass)
	labels=Modshogun::Labels.new
	labels.set_features(label_train_twoclass)

# *** 	svm=LibSVM(C, kernel, labels)
	svm=Modshogun::LibSVM.new
	svm.set_features(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train()

	kernel.init(feats_train, feats_test)
	labels = svm.apply().get_labels()
	supportvectors = sv_idx=svm.get_support_vectors()
	alphas=svm.get_alphas()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()


end
if __FILE__ == $0
	puts 'LibSVM'
	classifier_libsvm_modular(*parameter_list[0])

end
