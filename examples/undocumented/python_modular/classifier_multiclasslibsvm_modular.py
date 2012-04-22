from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_multiclasslibsvm_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,width=2.1,C=1,epsilon=1e-5):
	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Classifier import MulticlassLibSVM

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=Labels(label_train_multiclass)

	svm=MulticlassLibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train()

	kernel.init(feats_train, feats_test)
	out = svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	from sys import argv
	from numpy import save, load

	print('MulticlassLibSVM')
	[predictions, svm, labels] = classifier_multiclasslibsvm_modular(*parameter_list[0])
	if len(argv) > 2:
		if argv[1] == 'save':
			print('Save prediction to %s for future regression test' % argv[2])
			save(argv[2], labels)
		elif argv[1] == 'regression':
			print('Regression test from %s' % argv[2])
			labels_reg = load(argv[2])
			if (labels == labels_reg).all():
				print('Regression test passed!')
			else:
				print('Regression test FAILED!!!')

