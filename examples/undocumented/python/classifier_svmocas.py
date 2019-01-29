#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-5,1],[traindat,testdat,label_traindat,0.8,1e-5,1]]

def classifier_svmocas (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,C=0.9,epsilon=1e-5,num_threads=1):
	from shogun import BinaryLabels
	try:
		from shogun import SVMOcas
	except ImportError:
		print("SVMOcas not available")
		return
	import shogun as sg

	feats_train=sg.features(sg.csv_file(train_fname))
	feats_test=sg.features(sg.csv_file(test_fname))
	labels=BinaryLabels(sg.csv_file(label_fname))

	svm=SVMOcas(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.train()

	bias=svm.get_bias()
	w=svm.get_w()
	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('SVMOcas')
	classifier_svmocas(*parameter_list[0])
