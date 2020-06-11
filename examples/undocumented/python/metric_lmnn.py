#!/usr/bin/env python

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,3]]

def metric_lmnn(train_fname=traindat,test_fname=testdat,label_train_fname=label_traindat,k=3):
	try:
		from shogun import MulticlassLabels,LMNN,KNN
	except ImportError:
		return
	import shogun as sg

	# wrap features and labels into Shogun objects
	feats_train=sg.create_features(sg.read_csv(train_fname))
	feats_test=sg.create_features(sg.read_csv(test_fname))
	labels=MulticlassLabels(sg.read_csv(label_train_fname))

	# LMNN
	lmnn=LMNN(feats_train,labels,k)
	lmnn.train()
	lmnn_distance=lmnn.get_distance()

	# perform classification with KNN
	knn=KNN(k,lmnn_distance)
	knn.train(feats_train, labels)
	output=knn.apply(feats_test).get_labels()

	return lmnn,output

if __name__=='__main__':
	print('LMNN')
	metric_lmnn(*parameter_list[0])
