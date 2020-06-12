#!/usr/bin/env python

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list= [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]


def kernel_combined_custom_poly (train_fname = traindat,test_fname = testdat,train_label_fname=label_traindat):
    from shogun import CombinedFeatures, BinaryLabels
    from shogun import CustomKernel
    import shogun as sg

    kernel = sg.create_kernel("CombinedKernel")
    feats_train = CombinedFeatures()

    tfeats = sg.create_features(sg.read_csv(train_fname))
    tkernel = sg.create_kernel("PolyKernel", cache_size=10, degree=3)
    tkernel.init(tfeats, tfeats)
    K = tkernel.get_kernel_matrix()
    kernel.add("kernel_array", CustomKernel(K))

    subkfeats_train = sg.create_features(sg.read_csv(train_fname))
    feats_train.append_feature_obj(subkfeats_train)
    subkernel = sg.create_kernel("PolyKernel", cache_size=10, degree=2)
    kernel.add("kernel_array", subkernel)

    kernel.init(feats_train, feats_train)

    labels = BinaryLabels(sg.read_csv(train_label_fname))
    svm = sg.create_machine("LibSVM", C1=1.0, C2=1.0, kernel=kernel, labels=labels)
    svm.train()

    kernel = sg.create_kernel("CombinedKernel")
    feats_pred = CombinedFeatures()

    pfeats = sg.create_features(sg.read_csv(test_fname))
    tkernel = sg.create_kernel("PolyKernel", cache_size=10, degree=3)
    tkernel.init(tfeats, pfeats)
    K = tkernel.get_kernel_matrix()
    kernel.add("kernel_array", CustomKernel(K))

    subkfeats_test = sg.create_features(sg.read_csv(test_fname))
    feats_pred.append_feature_obj(subkfeats_test)
    subkernel = sg.create_kernel("PolyKernel", cache_size=10, degree=2)
    kernel.add("kernel_array", subkernel)
    kernel.init(feats_train, feats_pred)

    svm.put("kernel", kernel)
    svm.apply()
    km_train=kernel.get_kernel_matrix()
    return km_train,kernel

if __name__=='__main__':
    kernel_combined_custom_poly(*parameter_list[0])
