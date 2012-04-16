from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1,2000],[traindat,testdat,label_traindat,3,1,5000]]

def classifier_larank_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,C=0.9,num_threads=1,num_iter=5):
	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Classifier import LaRank
	from shogun.Mathematics import Math_init_random
	Math_init_random(17)

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	epsilon=1e-5
	labels=Labels(label_train_multiclass)

	svm=LaRank(C, kernel, labels)
	#svm.set_tau(1e-3)
	svm.set_batch_mode(False)
	#svm.io.enable_progress()
	svm.set_epsilon(epsilon)
	svm.train()
	out=svm.apply(feats_train).get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()


if __name__=='__main__':
	from sys import argv
	from numpy import save, load

	print('LaRank')
	[predictions, svm, labels] = classifier_larank_modular(*parameter_list[0])
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

