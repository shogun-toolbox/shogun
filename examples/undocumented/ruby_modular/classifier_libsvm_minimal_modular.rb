require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1]]

def classifier_libsvm_minimal_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,width=2.1,c=1)

	feats_train = Modshogun::RealFeatures.new 
	feats_train.set_feature_matrix(fm_train_real)
	feats_test = Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)
	
	kernel = Modshogun::GaussianKernel.new feats_train, feats_train, width

	labels = Modshogun::BinaryLabels.new label_train_twoclass
	svm = Modshogun::LibSVM.new c, kernel, labels
	svm.train

	kernel.init feats_train, feats_test
	out = svm.apply.get_labels()
	
	return out
end

if __FILE__ == $0
	puts 'LibSVM Minimal'
	pp classifier_libsvm_minimal_modular(*parameter_list[0])
end
