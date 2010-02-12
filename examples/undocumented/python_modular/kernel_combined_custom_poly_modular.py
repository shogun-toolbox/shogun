def combined_custom():
    from shogun.Features import CombinedFeatures, RealFeatures, Labels
    from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
    from shogun.Classifier import LibSVM

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

if __name__=='__main__':
    from tools.load import LoadMatrix
    lm=LoadMatrix()
    fm_train_real = lm.load_numbers('../data/fm_train_real.dat')
    fm_test_real = lm.load_numbers('../data/fm_test_real.dat')
    fm_label_twoclass = lm.load_labels('../data/label_train_twoclass.dat')
    combined_custom()
