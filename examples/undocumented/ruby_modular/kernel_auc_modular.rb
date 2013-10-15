require 'load'
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')
parameter_list = [[traindat,testdat,1.7], [traindat,testdat,1.6]]


def kernel_auc_modular(fm_train_real=traindat,label_train_real=testdat,width=1.7)

# ***	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)

# ***	subkernel=GaussianKernel(feats_train, feats_train, width)
	subkernel=Modshogun::GaussianKernel.new(feats_train, feats_train, width)

# ***	kernel=AUCKernel(0, subkernel)
	kernel=Modshogun::AUCKernel.new(0, subkernel)
	kernel.setup_auc_maximization( Modshogun::BinaryLabels.new(label_train_real) )
	km_train=kernel.get_kernel_matrix()
	return kernel
end

if __FILE__ == $0
	puts 'AUC'
	pp kernel_auc_modular(*parameter_list[0])
end
