from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,1.,1000,1],[traindat,testdat,label_traindat,1.,1000,1]]

def classifier_averaged_perceptron_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,learn_rate=1.,max_iter=1000,num_threads=1):
	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import AveragedPerceptron

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=Labels(label_train_twoclass)

	perceptron=AveragedPerceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	# only guaranteed to converge for separable data
	perceptron.train()

	perceptron.set_features(feats_test)
	out_labels = perceptron.apply().get_labels()
	return perceptron, out_labels

if __name__=='__main__':
	print 'AveragedPerceptron'
	classifier_averaged_perceptron_modular(*parameter_list[0])
