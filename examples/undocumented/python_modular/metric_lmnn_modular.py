#!/usr/bin/env python

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,3]]

def metric_lmnn_modular(train_fname=traindat,test_fname=testdat,label_train_fname=label_traindat,k=3):
	from modshogun import RealFeatures,MulticlassLabels,LMNN,KNN,CSVFile

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=MulticlassLabels(CSVFile(label_train_fname))

	# LMNN
	lmnn=LMNN(feats_train,labels,k)
	lmnn.train()
	lmnn_distance=lmnn.get_distance()

	# perform classification with KNN
	knn=KNN(k,lmnn_distance,labels)
	knn.train()
	output=knn.apply(feats_test).get_labels()

	return lmnn,output

if __name__=='__main__':
	print('LMNN')
	metric_lmnn_modular(*parameter_list[0])
