def larank ():
	print 'LaRank'

	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Classifier import LaRank

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=1
	epsilon=1e-5
	labels=Labels(label_train_multiclass)

	svm=LaRank(C, kernel, labels)
	#svm.set_tau(1e-3)
	#svm.set_batch_mode(False)
	#svm.io.enable_progress()
	svm.set_epsilon(epsilon)
	svm.train()
	out=svm.classify(feats_train).get_labels()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_multiclass=lm.load_labels('../data/label_train_multiclass.dat')
	larank()
