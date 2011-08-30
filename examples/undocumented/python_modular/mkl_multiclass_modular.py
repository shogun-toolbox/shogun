from tools.load import LoadMatrix
lm = LoadMatrix()
fm_train_real = lm.load_numbers('../data/fm_train_real.dat')
fm_test_real = lm.load_numbers('../data/fm_test_real.dat')
label_train_multiclass = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list=[
		[ fm_train_real, fm_test_real, label_train_multiclass, 1.2, 1.2, 1e-5, 1, 0.001, 1.5],
		[ fm_train_real, fm_test_real, label_train_multiclass, 5, 1.2, 1e-2, 1, 0.001, 2]]

def mkl_multiclass_modular(fm_train_real, fm_test_real, label_train_multiclass, width, c, epsilon, num_threads, mkl_epsilon, mkl_norm)

	from shogun.Features import CombinedFeatures, RealFeatures, Labels
	from shogun.Kernel import CombinedKernel, GaussianKernel, LinearKernel,PolyKernel
	from shogun.Classifier import MKLMultiClass

	kernel = Modshogun::CombinedKernel.new
	feats_train = Modshogun::CombinedFeatures.new
	feats_test = Modshogun::CombinedFeatures.new

	subkfeats_train = Modshogun::RealFeatures.new
	subkfeats_train.set_feature_matrix(fm_train_real)
	subkfeats_test = Modshogun::RealFeatures.new
	subkfeats_test.set_feature_matrix(fm_test_real)
	subkernel = Modshogun::GaussianKernel.new(10, width)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train = Modshogun::RealFeatures.new
	subkfeats_train.set_feature_matrix(fm_train_real)
	subkfeats_test = Modshogun::RealFeatures.new
	subkfeats_test.set_feature_matrix(fm_test_real)
	subkernel = Modshogun::LinearKernel.new
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train = Modshogun::RealFeatures.new
	subkfeats_train.set_feature_matrix(fm_train_real)
	subkfeats_test = Modshogun::RealFeatures.new
	subkfeats_test.set_feature_matrix(fm_test_real)
	subkernel = Modshogun::PolyKernel.new(10,2)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)
	
	kernel.init(feats_train, feats_train)

	labels = Modshogun::Labels.new(label_train_multiclass)

	mkl = Modshogun::MKLMultiClass.new(c, kernel, labels)
	
	mkl.set_epsilon(epsilon);
	mkl.parallel.set_num_threads(num_threads)
	mkl.set_mkl_epsilon(mkl_epsilon)
	mkl.set_mkl_norm(mkl_norm)

	mkl.train

	kernel.init(feats_train, feats_test)

	out =  mkl.apply.get_labels()
	return out

if __name__ == '__main__':
	print 'mkl_multiclass'
	mkl_multiclass_modular(*parameter_list[0])
