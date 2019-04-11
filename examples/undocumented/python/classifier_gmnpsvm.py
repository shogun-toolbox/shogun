#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_gmnpsvm (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,width=2.1,C=1,epsilon=1e-5):
	from shogun import MulticlassLabels
	from shogun import GMNPSVM, CSVFile
	import shogun as sg

	feats_train=sg.features(CSVFile(train_fname))
	feats_test=sg.features(CSVFile(test_fname))
	labels=MulticlassLabels(CSVFile(label_fname))

	kernel=sg.kernel("GaussianKernel", log_width=width)

	svm=GMNPSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train(feats_train)

	out=svm.apply(feats_test).get_labels()
	return out,kernel
if __name__=='__main__':
	print('GMNPSVM')
	classifier_gmnpsvm(*parameter_list[0])
