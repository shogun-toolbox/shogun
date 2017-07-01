#!/usr/bin/env python
from numpy import array

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False, False])

parameter_list = [[traindat,testdat,label_traindat,feattypes]]

def multiclass_randomforest_modular(train=traindat,test=testdat,labels=label_traindat,ft=feattypes):
	try:
		from modshogun import RealFeatures, MulticlassLabels, CSVFile, RandomForest, MajorityVote
	except ImportError:
		print("Could not import Shogun modules")
		return

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(CSVFile(train))
	feats_test=RealFeatures(CSVFile(test))
	train_labels=MulticlassLabels(CSVFile(labels))

	# Random Forest formation
	rand_forest=RandomForest(feats_train,train_labels,20,1)
	rand_forest.set_feature_types(ft)
	rand_forest.set_combination_rule(MajorityVote())
	rand_forest.train()

	# Classify test data
	output=rand_forest.apply_multiclass(feats_test).get_labels()

	return rand_forest,output

if __name__=='__main__':
	print('RandomForest')
	multiclass_randomforest_modular(*parameter_list[0])
