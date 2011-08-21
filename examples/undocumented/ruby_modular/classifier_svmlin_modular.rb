# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-5,1],[traindat,testdat,label_traindat,0.8,1e-5,1]]

def classifier_svmlin_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,C=0.9,epsilon=1e-5,num_threads=1)

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

# *** 	svm=SVMLin(C, feats_train, labels)
	svm=Modshogun::SVMLin.new
	svm.set_features(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.get_bias()
	svm.get_w()
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()



end
if __FILE__ == $0
	puts 'SVMLin'
	classifier_svmlin_modular(*parameter_list[0])

end
