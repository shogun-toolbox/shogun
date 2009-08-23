def subgradient_svm ():
	print 'SubGradientSVM'

	from shogun.Features import RealFeatures, SparseRealFeatures, Labels
	from shogun.Classifier import SubGradientSVM

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-3
	num_threads=1
	max_train_time=1.
	labels=Labels(label_train_twoclass)

	svm=SubGradientSVM(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.set_max_train_time(max_train_time)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
	subgradient_svm()
