#!/usr/bin/env python
from numpy import array

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False, False])

parameter_list = [[traindat,testdat,label_traindat,feattypes]]

def multiclass_cartree_modular(train=traindat,test=testdat,labels=label_traindat,ft=feattypes):
	try:
		from modshogun import RealFeatures, MulticlassLabels, CSVFile, CARTree, PT_MULTICLASS
	except ImportError:
		print("Could not import Shogun modules")
		return

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(CSVFile(train))
	feats_test=RealFeatures(CSVFile(test))
	train_labels=MulticlassLabels(CSVFile(labels))

	# CART Tree formation with 5 fold cross-validation pruning
	c=CARTree(ft,PT_MULTICLASS,5,True)
	c.set_labels(train_labels)
	c.train(feats_train)

	# Classify test data
	output=c.apply_multiclass(feats_test).get_labels()

	return c,output

if __name__=='__main__':
	print('CARTree')
	multiclass_cartree_modular(*parameter_list[0])
