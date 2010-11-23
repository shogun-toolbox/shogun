from tools.load import LoadMatrix
lm=LoadMatrix()

parameter_list= [[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),lm.load_labels('../data/label_train_twoclass.dat')],[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),lm.load_labels('../data/label_train_twoclass.dat')]]


def kernel_combined_custom_poly_modular(fm_train_real = lm.load_numbers('../data/fm_train_real.dat'),fm_test_real = lm.load_numbers('../data/fm_test_real.dat'),fm_label_twoclass=lm.load_labels('../data/label_train_twoclass.dat')):
    from shogun.Features import CombinedFeatures, RealFeatures, Labels
    from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
    from shogun.Classifier import LibSVM
    fm_train_real       = fm_train_real
    fm_test_real        = fm_test_real
    fm_label_twoclass   =  fm_label_twoclass
    kernel = CombinedKernel()
    feats_train = CombinedFeatures()
    
    tfeats = RealFeatures(fm_train_real)
    tkernel = PolyKernel(10,3)
    tkernel.init(tfeats, tfeats)
    K = tkernel.get_kernel_matrix()
    kernel.append_kernel(CustomKernel(K))
        
    subkfeats_train = RealFeatures(fm_train_real)
    feats_train.append_feature_obj(subkfeats_train)
    subkernel = PolyKernel(10,2)
    kernel.append_kernel(subkernel)

    kernel.init(feats_train, feats_train)
    
    labels = Labels(fm_label_twoclass)
    svm = LibSVM(1.0, kernel, labels)
    svm.train()

    kernel = CombinedKernel()
    feats_pred = CombinedFeatures()

    pfeats = RealFeatures(fm_test_real)
    tkernel = PolyKernel(10,3)
    tkernel.init(tfeats, pfeats)
    K = tkernel.get_kernel_matrix()
    kernel.append_kernel(CustomKernel(K))

    subkfeats_test = RealFeatures(fm_test_real)
    feats_pred.append_feature_obj(subkfeats_test)
    subkernel = PolyKernel(10, 2)
    kernel.append_kernel(subkernel)
    kernel.init(feats_train, feats_pred)

    svm.set_kernel(kernel)
    svm.classify()
    km_train=kernel.get_kernel_matrix()
    print km_train

if __name__=='__main__':
    from tools.load import LoadMatrix
    lm=LoadMatrix()
    fm_train_real = lm.load_numbers('../data/fm_train_real.dat')
    fm_test_real = lm.load_numbers('../data/fm_test_real.dat')
    fm_label_twoclass = lm.load_labels('../data/label_train_twoclass.dat')
    kernel_combined_custom_poly_modular(*parameter_list[0])
