#!/usr/bin/env python

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,3]]

def classifier_knn_modular(train_fname=traindat,test_fname=testdat,label_train_fname=label_traindat, k=3):
	from modshogun import RealFeatures, MulticlassLabels, KNN, EuclideanDistance, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	distance=EuclideanDistance(feats_train, feats_train)

	labels=MulticlassLabels(CSVFile(label_train_fname))

	knn=KNN(k, distance, labels)
	knn_train = knn.train()
	output=knn.apply(feats_test).get_labels()
	multiple_k=knn.classify_for_multiple_k()

	return knn,knn_train,output,multiple_k

if __name__=='__main__':
	print('KNN')
	classifier_knn_modular(*parameter_list[0])
