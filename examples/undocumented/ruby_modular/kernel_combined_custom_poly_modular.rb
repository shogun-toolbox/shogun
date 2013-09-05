require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list= [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]


def kernel_combined_custom_poly_modular(fm_train_real = traindat,fm_test_real = testdat,fm_label_twoclass=label_traindat)
   
    kernel = Modshogun::CombinedKernel.new
    feats_train = Modshogun::CombinedFeatures.new
    
    tfeats = Modshogun::RealFeatures.new
    tfeats.set_feature_matrix(fm_train_real)
    tkernel = Modshogun::PolyKernel.new(10,3)
    tkernel.init(tfeats, tfeats)
    k = tkernel.get_kernel_matrix()
    f = Modshogun::CustomKernel.new
    f.set_full_kernel_matrix_from_full(k)
    kernel.append_kernel(f)
        
    subkfeats_train = Modshogun::RealFeatures.new
    subkfeats_train.set_feature_matrix(fm_train_real)
    feats_train.append_feature_obj(subkfeats_train)
    subkernel = Modshogun::PolyKernel.new(10,2)
    kernel.append_kernel(subkernel)

    kernel.init(feats_train, feats_train)
    
    labels = Modshogun::BinaryLabels.new(fm_label_twoclass)
    svm = Modshogun::LibSVM.new(1.0, kernel, labels)
    svm.train()

    kernel = Modshogun::CombinedKernel.new
    feats_pred = Modshogun::CombinedFeatures.new

    pfeats = Modshogun::RealFeatures.new
    pfeats.set_feature_matrix(fm_test_real)
    tkernel = Modshogun::PolyKernel.new(10,3)
    tkernel.init(tfeats, pfeats)
    k = tkernel.get_kernel_matrix()
    f = Modshogun::CustomKernel.new
    f.set_full_kernel_matrix_from_full(k)
    kernel.append_kernel(f)

    subkfeats_test = Modshogun::RealFeatures.new
    subkfeats_test.set_feature_matrix(fm_test_real)
    feats_pred.append_feature_obj(subkfeats_test)
    subkernel = Modshogun::PolyKernel.new(10, 2)
    kernel.append_kernel(subkernel)
    kernel.init(feats_train, feats_pred)

    svm.set_kernel(kernel)
    svm.apply()
    km_train=kernel.get_kernel_matrix()
    return km_train,kernel
end

if __FILE__ == $0
    puts 'Combined Custom Poly Modular'
    pp kernel_combined_custom_poly_modular(*parameter_list[0])
end
