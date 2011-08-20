# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,1.1,1e-5,1],[traindat,testdat,label_traindat,1.2,1e-5,1]]

def classifier_svmlight_modular(fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,C=1.2,epsilon=1e-5,num_threads=1)
	try:
	except ImportError:
	puts 'No support for SVMLight available.'
		return

# *** 	feats_train=StringCharFeatures(DNA)
	feats_train=Modshogun::StringCharFeatures.new
	feats_train.set_features(DNA)
	feats_train.set_features(fm_train_dna)
# *** 	feats_test=StringCharFeatures(DNA)
	feats_test=Modshogun::StringCharFeatures.new
	feats_test.set_features(DNA)
	feats_test.set_features(fm_test_dna)
	degree=20

# *** 	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)
	kernel=Modshogun::WeightedDegreeStringKernel.new
	kernel.set_features(feats_train, feats_train, degree)

# *** 	labels=Labels(label_train_dna)
	labels=Modshogun::Labels.new
	labels.set_features(label_train_dna)

# *** 	svm=SVMLight(C, kernel, labels)
	svm=Modshogun::SVMLight.new
	svm.set_features(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.apply().get_labels()
	return kernel

end
if __FILE__ == $0
	puts 'SVMLight'
	classifier_svmlight_modular(*parameter_list[0])

end
