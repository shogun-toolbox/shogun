#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
label_traindat = '../data/label_train_twoclass.dat'
parameter_list = [[traindat,label_traindat,1.7], [traindat,label_traindat,1.6]]

def kernel_auc_modular (train_fname=traindat,label_fname=label_traindat,width=1.7):
	from modshogun import GaussianKernel, AUCKernel, RealFeatures
	from modshogun import BinaryLabels, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	subkernel=GaussianKernel(feats_train, feats_train, width)

	kernel=AUCKernel(0, subkernel)
	kernel.setup_auc_maximization(BinaryLabels(CSVFile(label_fname)))
	km_train=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('AUC')
	kernel_auc_modular(*parameter_list[0])
