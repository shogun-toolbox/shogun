require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_gmnpsvm_modular(fm_train_real, fm_test_real, label_train_multiclass, width, c, epsilon)

	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)

	kernel=Modshogun::GaussianKernel.new(feats_train, feats_train, width)

	labels=Modshogun::Labels.new
	labels.set_labels(label_train_multiclass)

	svm=Modshogun::GMNPSVM.new(c, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train(feats_train)
	kernel.init(feats_train, feats_test)
	out=svm.apply(feats_test).get_labels()
	return out,kernel
end

if __FILE__ == $0
	puts 'GMNPSVM'
	pp classifier_gmnpsvm_modular(*parameter_list[0])
end
