#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_gpbtsvm (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,width=2.1,C=1,epsilon=1e-5):
	from shogun import RealFeatures, BinaryLabels
	from shogun import CSVFile
	import shogun as sg
	try:
		from shogun import GPBTSVM
	except ImportError:
		print("GPBTSVM not available")
		exit(0)

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))
	kernel=sg.kernel("GaussianKernel", log_width=width)

	svm=GPBTSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train(feats_train)

	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get_labels()


if __name__=='__main__':
	print('GPBTSVM')
	classifier_gpbtsvm(*parameter_list[0])
