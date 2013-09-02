#!/usr/bin/env python

from tools.load import LoadMatrix

# load training data and test features
lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,3],[traindat,testdat,label_traindat,3]]

def metric_lmnn_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,k=3):
	from modshogun import RealFeatures,MulticlassLabels,LMNN,KNN

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	labels=MulticlassLabels(label_train_multiclass)

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
