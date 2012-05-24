require 'modshogun'
require 'load'

traindat = load_numbers('../data/fm_train_real.dat')
testdat = load_numbers('../data/fm_test_real.dat')
label_traindat = load_labels('../data/label_train_twoclass.dat')

parameter_list = {{traindat,testdat,label_traindat,1.,1000,1},{traindat,testdat,label_traindat,1.,100,1}}

function classifier_averaged_perceptron_modular (fm_train_real,fm_test_real,label_train_twoclass,learn_rate,max_iter,num_threads)

	feats_train=modshogun.RealFeatures(fm_train_real)
	feats_test=modshogun.RealFeatures(fm_test_real)

	labels=modshogun.BinaryLabels(label_train_twoclass)

	perceptron=modshogun.AveragedPerceptron(feats_train, labels)
	perceptron:set_learn_rate(learn_rate)
	perceptron:set_max_iter(max_iter)

	perceptron:train()

	perceptron:set_features(feats_test)
	out_labels = perceptron:apply():get_labels()

	return perceptron, out_labels
end

if debug.getinfo(3) == nill then
	print 'AveragedPerceptron'
	classifier_averaged_perceptron_modular(unpack(parameter_list[1]))
end
