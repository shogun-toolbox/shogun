#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_gmnpsvm_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,width=2.1,C=1,epsilon=1e-5):

	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import GaussianKernel
	from modshogun import GMNPSVM

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=MulticlassLabels(label_train_multiclass)

	svm=GMNPSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train(feats_train)
	kernel.init(feats_train, feats_test)
	out=svm.apply(feats_test).get_labels()
	return out,kernel
if __name__=='__main__':
	print('GMNPSVM')
	classifier_gmnpsvm_modular(*parameter_list[0])
