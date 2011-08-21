# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-3],[traindat,testdat,label_traindat,0.8,1e-2]]

def classifier_liblinear_modular(fm_train_real, fm_test_real,
		label_train_twoclass, C, epsilon):

	Math_init_random(17)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)
# *** 	labels=Labels(label_train_twoclass)
	labels=Modshogun::Labels.new
	labels.set_features(label_train_twoclass)

# *** 	svm=LibLinear(C, feats_train, labels)
	svm=Modshogun::LibLinear.new
	svm.set_features(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()




end
if __FILE__ == $0
	puts 'LibLinear'
	classifier_liblinear_modular(*parameter_list[0])



end
