from shogun.Features import CombinedFeatures, RealFeatures, Labels
from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
from shogun.Classifier import MKLClassification

def combined_custom():


    ##################################
    # set up and train

    # create some poly train/test matrix
    tfeats = RealFeatures(fm_train_real)
    tkernel = PolyKernel(10,3)
    tkernel.init(tfeats, tfeats)
    K_train = tkernel.get_kernel_matrix()

    pfeats = RealFeatures(fm_test_real)
    tkernel.init(tfeats, pfeats)
    K_test = tkernel.get_kernel_matrix()

    # create combined train features
    feats_train = CombinedFeatures()
    feats_train.append_feature_obj(RealFeatures(fm_train_real))

    # and corresponding combined kernel
    kernel = CombinedKernel()
    kernel.append_kernel(CustomKernel(K_train))
    kernel.append_kernel(PolyKernel(10,2))
    kernel.init(feats_train, feats_train)

    # train mkl
    labels = Labels(fm_label_twoclass)
    mkl = MKLClassification()

    # which norm to use for MKL
    mkl.set_mkl_norm(1) #2,3

    # set cost (neg, pos)
    mkl.set_C(1, 1)

    # set kernel and labels
    mkl.set_kernel(kernel)
    mkl.set_labels(labels)

    # train
    mkl.train()
    #w=kernel.get_subkernel_weights()
    #kernel.set_subkernel_weights(w)


    ##################################
    # test

    # create combined test features
    feats_pred = CombinedFeatures()
    feats_pred.append_feature_obj(RealFeatures(fm_test_real))

    # and corresponding combined kernel
    kernel = CombinedKernel()
    kernel.append_kernel(CustomKernel(K_test))
    kernel.append_kernel(PolyKernel(10, 2))
    kernel.init(feats_train, feats_pred)

    # and classify
    mkl.set_kernel(kernel)
    mkl.classify()


if __name__=='__main__':
    from tools.load import LoadMatrix
    lm = LoadMatrix()
    fm_train_real = lm.load_numbers('../data/fm_train_real.dat')
    fm_test_real = lm.load_numbers('../data/fm_test_real.dat')
    fm_label_twoclass = lm.load_labels('../data/label_train_twoclass.dat')
    fm_train_real.shape
    fm_test_real.shape
    combined_custom()

