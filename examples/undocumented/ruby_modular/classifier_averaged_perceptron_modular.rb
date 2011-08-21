require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [traindat,testdat,label_traindat,2,1000,1]

def classifier_averaged_perceptron_modular(fm_train_real, fm_test_real, label_train_twoclass, learn_rate, max_iter, num_threads)

	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)

	labels=Modshogun::Labels.new
	labels.set_labels(label_train_twoclass)

	perceptron=Modshogun::AveragedPerceptron.new(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	# only guaranteed to converge for separable data
	perceptron.train()

	perceptron.set_features(feats_test)
	out_labels = perceptron.apply.get_labels
	#puts out_labels
	return perceptron, out_labels
end

if __FILE__ == $0
	puts 'AveragedPerceptron'
	pp classifier_averaged_perceptron_modular(*parameter_list)
end
