#!/usr/bin/env python
from numpy import array

# create data
train_data = array([[1.0, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0],
[2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0],
[3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0],
[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]])

train_labels = array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

test_data = array([[2.0, 2.0, 1.0, 3.0, 3.0],
[2.0, 1.0, 2.0, 1.0, 2.0],
[3.0, 2.0, 1.0, 3.0, 2.0],
[1.0, 2.0, 1.0, 2.0, 1.0]])

parameter_list = [[train_data, train_labels, test_data]]

def multiclass_id3classifiertree_modular(train=train_data,labels=train_labels,test=test_data):
	try:
		from modshogun import RealFeatures, MulticlassLabels, ID3ClassifierTree
	except ImportError:
		return

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(train)
	feats_test=RealFeatures(test)
	feats_labels=MulticlassLabels(labels)

	# ID3 Tree formation
	id3=ID3ClassifierTree()
	id3.set_labels(feats_labels)
	id3.train(feats_train)

	# Classify test data
	output=id3.apply_multiclass(feats_test).get_labels()

	return id3,output

if __name__=='__main__':
	print('ID3ClassifierTree')
	multiclass_id3classifiertree_modular(*parameter_list[0])
