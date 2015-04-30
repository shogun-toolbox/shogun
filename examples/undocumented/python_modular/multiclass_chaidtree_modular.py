#!/usr/bin/env python
from numpy import array, dtype, int32

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set both input attributes as continuous i.e. 2
feattypes = array([2, 2],dtype=int32)

parameter_list = [[traindat,testdat,label_traindat,feattypes]]

def multiclass_chaidtree_modular(train=traindat,test=testdat,labels=label_traindat,ft=feattypes):
	try:
		from modshogun import RealFeatures, MulticlassLabels, CSVFile, CHAIDTree
	except ImportError:
		print("Could not import Shogun modules")
		return

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(CSVFile(train))
	feats_test=RealFeatures(CSVFile(test))
	train_labels=MulticlassLabels(CSVFile(labels))

	# CHAID Tree formation with nominal dependent variable
	c=CHAIDTree(0,feattypes,10)
	c.set_labels(train_labels)
	c.train(feats_train)

	# Classify test data
	output=c.apply_multiclass(feats_test).get_labels()

	return c,output

if __name__=='__main__':
	print('CHAIDTree')
	multiclass_chaidtree_modular(*parameter_list[0])
