# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]
#    fm_train_real.shape
#    fm_test_real.shape
#    combined_custom()

def mkl_binclass_modular(fm_train_real=traindat,fm_test_real=testdat,fm_label_twoclass = label_traindat)

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
    mkl.apply()
    return mkl.apply(),kernel


end
if __FILE__ == $0
    mkl_binclass_modular (*parameter_list[0])

end
