def mkl_multiclass ():
	print 'mkl_multiclass'

	from shogun.Features import RealFeatures, Labels
	from shogun.Kernel import GaussianKernel
	from shogun.Kernel import LinearKernel
	from shogun.Kernel import Chi2Kernel
	from shogun.Classifier.MKL import MKLMultiClass



	kernel=CombinedKernel()
	feats_train=CombinedFeatures()
	feats_test=CombinedFeatures()

	subkfeats_train=RealFeatures(fm_train_real)
	subkfeats_test=RealFeatures(fm_test_real)
	subkernel=GaussianKernel(10, 1.2)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)


	subkfeats_train=RealFeatures(fm_train_real)
	subkfeats_test=RealFeatures(fm_test_real)
	subkernel=LinearKernel(10)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)


	subkfeats_train=RealFeatures(fm_train_real)
	subkfeats_test=RealFeatures(fm_test_real)
	subkernel=Chi2Kernel(10,1.2)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)
	kernel.init(feats_train, feats_train)

	C=1
	epsilon=1e-5
	labels=Labels(label_train_multiclass)

	svm=MKLMultiClass(C, kernel, labels)
	svm.set_epsilon(epsilon);
	svm.parallel.set_num_threads(num_threads)

	mkl_eps=0.01
	mkl_norm=1
	svm.set_mkl_parameters(mkl_eps,0,mkl_norm)
	svm.train(feats_train)
	#kernel.init(feats_train, feats_test)
	svm.classify(feats_test).get_labels()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	label_train_multiclass=lm.load_labels('../data/label_train_multiclass.dat')
	gmnpsvm()
