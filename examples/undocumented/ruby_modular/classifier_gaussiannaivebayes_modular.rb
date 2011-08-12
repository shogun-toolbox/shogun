# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def classifier_gaussiannaivebayes_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat)

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	labels=Labels(label_train_multiclass)

	gnb=GaussianNaiveBayes(feats_train, labels)
	gnb_train = gnb.train()
	output=gnb.apply(feats_test).get_labels()
	return gnb, gnb_train, output


end
if __FILE__ == $0
	print 'GaussianNaiveBayes'
	classifier_gaussiannaivebayes_modular(*parameter_list[0])

end
