# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
lm = LoadMatrix()
fm_train_real = LoadMatrix.load_numbers('../data/fm_train_real.dat')
fm_test_real = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_train_multiclass = LoadMatrix.load_labels('../data/label_train_multiclass.dat')

parameter_list=[
		[ fm_train_real, fm_test_real, label_train_multiclass, 1.2, 1.2, 1e-5, 1, 0.001, 1.5],
		[ fm_train_real, fm_test_real, label_train_multiclass, 5, 1.2, 1e-2, 1, 0.001, 2]]

def mkl_multiclass_modular(fm_train_real, fm_test_real, label_train_multiclass,
	width, C, epsilon, num_threads, mkl_epsilon, mkl_norm):


	kernel = CombinedKernel()
	feats_train = CombinedFeatures()
	feats_test = CombinedFeatures()

	subkfeats_train = RealFeatures(fm_train_real)
	subkfeats_test = RealFeatures(fm_test_real)
	subkernel = GaussianKernel(10, width)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train = RealFeatures(fm_train_real)
	subkfeats_test = RealFeatures(fm_test_real)
	subkernel = LinearKernel()
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train = RealFeatures(fm_train_real)
	subkfeats_test = RealFeatures(fm_test_real)
	subkernel = PolyKernel(10,2)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)
	
	kernel.init(feats_train, feats_train)

	labels = Labels(label_train_multiclass)

	mkl = MKLMultiClass(C, kernel, labels)
	
	mkl.set_epsilon(epsilon);
	mkl.parallel.set_num_threads(num_threads)
	mkl.set_mkl_epsilon(mkl_epsilon)
	mkl.set_mkl_norm(mkl_norm)

	mkl.train()

	kernel.init(feats_train, feats_test)

	out =  mkl.apply().get_labels()
	return out


end
if __name__ == '__main__':
	print 'mkl_multiclass'
	mkl_multiclass_modular(*parameter_list[0])

end
