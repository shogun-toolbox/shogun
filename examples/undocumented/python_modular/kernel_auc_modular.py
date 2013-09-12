#!/usr/bin/env python
traindat = double(lm.load_numbers('../data/fm_train_real.dat'))
testdat = lm.load_labels('../data/label_train_twoclass.dat')
parameter_list = [[traindat,testdat,1.7], [traindat,testdat,1.6]]

def kernel_auc_modular (fm_train_real=traindat,label_train_real=testdat,width=1.7):
	from modshogun import GaussianKernel, AUCKernel
	from modshogun import RealFeatures, BinaryLabels

	feats_train=RealFeatures(fm_train_real)

	subkernel=GaussianKernel(feats_train, feats_train, width)

	kernel=AUCKernel(0, subkernel)
	kernel.setup_auc_maximization( BinaryLabels(label_train_real) )
	km_train=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('AUC')
	kernel_auc_modular(*parameter_list[0])
