#!/usr/bin/env python

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list= [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]


def kernel_combined_custom_poly (train_fname = traindat,test_fname = testdat,train_label_fname=label_traindat):
    from shogun import CombinedFeatures, RealFeatures, BinaryLabels
    from shogun import CombinedKernel, PolyKernel, CustomKernel
    from shogun import CSVFile
    import shogun as sg

    kernel = CombinedKernel()
    feats_train = CombinedFeatures()

    tfeats = RealFeatures(CSVFile(train_fname))
    tkernel = PolyKernel(10,3)
    tkernel.init(tfeats, tfeats)
    K = tkernel.get_kernel_matrix()
    kernel.append_kernel(CustomKernel(K))

    subkfeats_train = RealFeatures(CSVFile(train_fname))
    feats_train.append_feature_obj(subkfeats_train)
    subkernel = PolyKernel(10,2)
    kernel.append_kernel(subkernel)

    kernel.init(feats_train, feats_train)

    labels = BinaryLabels(CSVFile(train_label_fname))
    svm = sg.machine("LibSVM", C1=1.0, C2=1.0, kernel=kernel, labels=labels)
    svm.train()

    kernel = CombinedKernel()
    feats_pred = CombinedFeatures()

    pfeats = RealFeatures(CSVFile(test_fname))
    tkernel = PolyKernel(10,3)
    tkernel.init(tfeats, pfeats)
    K = tkernel.get_kernel_matrix()
    kernel.append_kernel(CustomKernel(K))

    subkfeats_test = RealFeatures(CSVFile(test_fname))
    feats_pred.append_feature_obj(subkfeats_test)
    subkernel = PolyKernel(10, 2)
    kernel.append_kernel(subkernel)
    kernel.init(feats_train, feats_pred)

    svm.put("kernel", kernel)
    svm.apply()
    km_train=kernel.get_kernel_matrix()
    return km_train,kernel

if __name__=='__main__':
    kernel_combined_custom_poly(*parameter_list[0])
