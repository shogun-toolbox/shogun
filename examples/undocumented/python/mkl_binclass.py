#!/usr/bin/env python
from shogun import CombinedFeatures, BinaryLabels
from shogun import CustomKernel
from shogun import MKLClassification
import shogun as sg
from tools.load import LoadMatrix
lm=LoadMatrix()

#only run example if SVMLight is included as LibSVM solver crashes in MKLClassification
try:
	from shogun import create_machine
	create_machine("SVMLight")
except SystemError:
	print("SVMLight not available")
	exit(0)

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]
#    fm_train_real.shape
#    fm_test_real.shape
#    combined_custom()

def mkl_binclass (fm_train_real=traindat,fm_test_real=testdat,fm_label_twoclass = label_traindat):

    ##################################
    # set up and train

    # create some poly train/test matrix
    tfeats = sg.create_features(fm_train_real)
    tkernel = sg.create_kernel("PolyKernel", cache_size=10, degree=3)
    tkernel.init(tfeats, tfeats)
    K_train = tkernel.get_kernel_matrix()

    pfeats = sg.create_features(fm_test_real)
    tkernel.init(tfeats, pfeats)
    K_test = tkernel.get_kernel_matrix()

    # create combined train features
    feats_train = CombinedFeatures()
    feats_train.append_feature_obj(sg.create_features(fm_train_real))

    # and corresponding combined kernel
    kernel = sg.create_kernel("CombinedKernel")
    kernel.add("kernel_array", CustomKernel(K_train))
    kernel.add("kernel_array", sg.create_kernel("PolyKernel", cache_size=10,
                                   degree=2))
    kernel.init(feats_train, feats_train)

    # train mkl
    labels = BinaryLabels(fm_label_twoclass)
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
    feats_pred.append_feature_obj(sg.create_features(fm_test_real))

    # and corresponding combined kernel
    kernel = sg.create_kernel("CombinedKernel")
    kernel.add("kernel_array", CustomKernel(K_test))
    kernel.add("kernel_array", sg.create_kernel("PolyKernel", cache_size=10, degree=2))
    kernel.init(feats_train, feats_pred)

    # and classify
    mkl.set_kernel(kernel)
    mkl.apply()
    return mkl.apply(),kernel

if __name__=='__main__':
    mkl_binclass (*parameter_list[0])
