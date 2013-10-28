#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,2.2,1,1e-7],[traindat,testdat,2.1,1,1e-5]]

def classifier_libsvmoneclass_modular (train_fname=traindat,test_fname=testdat,width=2.1,C=1,epsilon=1e-5):
	from modshogun import RealFeatures, GaussianKernel, LibSVMOneClass, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=GaussianKernel(feats_train, feats_train, width)

	svm=LibSVMOneClass(C, kernel)
	svm.set_epsilon(epsilon)
	svm.train()

	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('LibSVMOneClass')
	classifier_libsvmoneclass_modular(*parameter_list[0])
