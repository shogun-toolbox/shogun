require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,1,1000,1],[traindat,testdat,label_traindat,1,1000,1]]

def classifier_averaged_perceptron_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,learn_rate=1,max_iter=1000,num_threads=1)

	feats_train=Modshogun::RealFeatures.new(fm_train_real)
	feats_test=Modshogun::RealFeatures.new(fm_test_real)

	labels=Modshogun::Labels.new(label_train_twoclass)

	perceptron=Modshogun::AveragedPerceptron.new(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	# only guaranteed to converge for separable data
	perceptron.train()

	perceptron.set_features(feats_test)
	out_labels = perceptron.apply().get_labels()
	print out_labels
	return perceptron, out_labels
end

if __FILE__ == $0
	puts 'AveragedPerceptron'
	classifier_averaged_perceptron_modular(*parameter_list[0])
end
