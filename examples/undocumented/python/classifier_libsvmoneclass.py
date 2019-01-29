#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,2.2,1,1e-7],[traindat,testdat,2.1,1,1e-5]]

def classifier_libsvmoneclass (train_fname=traindat,test_fname=testdat,width=2.1,C=1,epsilon=1e-5):
	from shogun import RealFeatures, CSVFile
	import shogun as sg

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=sg.kernel("GaussianKernel", log_width=width)

	svm=sg.machine("LibSVMOneClass", C1=C, C2=C, kernel=kernel, epsilon=epsilon)
	svm.train(feats_train)

	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get("labels")

if __name__=='__main__':
	print('LibSVMOneClass')
	classifier_libsvmoneclass(*parameter_list[0])
