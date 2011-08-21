# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
# *** train=LoadMatrix.load_numbers('../data/fm_train_real.dat')
train=Modshogun::LoadMatrix.new
train.set_features.load_numbers('../data/fm_train_real.dat')
# *** test=LoadMatrix.load_numbers('../data/fm_test_real.dat')
test=Modshogun::LoadMatrix.new
test.set_features.load_numbers('../data/fm_test_real.dat')
# *** labels=LoadMatrix.load_labels('../data/label_train_twoclass.dat')
labels=Modshogun::LoadMatrix.new
labels.set_features.load_labels('../data/label_train_twoclass.dat')

parameter_list=[[train,test,labels,5,1e-3,3.0], [train,test,labels,0.9,1e-2,1.0]]

def classifier_subgradientsvm_modular(fm_train_real, fm_test_real,
		label_train_twoclass, C, epsilon, max_train_time):


# *** 	realfeat=RealFeatures(fm_train_real)
	realfeat=Modshogun::RealFeatures.new
	realfeat.set_features(fm_train_real)
# *** 	feats_train=SparseRealFeatures()
	feats_train=Modshogun::SparseRealFeatures.new
	feats_train.set_features()
	feats_train.obtain_from_simple(realfeat)
# *** 	realfeat=RealFeatures(fm_test_real)
	realfeat=Modshogun::RealFeatures.new
	realfeat.set_features(fm_test_real)
# *** 	feats_test=SparseRealFeatures()
	feats_test=Modshogun::SparseRealFeatures.new
	feats_test.set_features()
	feats_test.obtain_from_simple(realfeat)

# *** 	labels=Labels(label_train_twoclass)
	labels=Modshogun::Labels.new
	labels.set_features(label_train_twoclass)

# *** 	svm=SubGradientSVM(C, feats_train, labels)
	svm=Modshogun::SubGradientSVM.new
	svm.set_features(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.set_max_train_time(max_train_time)
	svm.train()

	svm.set_features(feats_test)
	labels=svm.apply().get_labels()

	return labels, svm


end
if __FILE__ == $0
	puts 'SubGradientSVM'
	classifier_subgradientsvm_modular(*parameter_list[0])

end
