#!/usr/bin/env python
from numpy import array

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False, False])

parameter_list = [[traindat,testdat,label_traindat,feattypes]]

def multiclass_c45classifiertree_modular(train=traindat,test=testdat,labels=label_traindat,ft=feattypes):
	try:
		from modshogun import RealFeatures, MulticlassLabels, CSVFile, C45ClassifierTree
		from numpy import random, int32
	except ImportError:
		print("Could not import Shogun and/or numpy modules")
		return

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(CSVFile(train))
	feats_test=RealFeatures(CSVFile(test))
	train_labels=MulticlassLabels(CSVFile(labels))

	# divide train dataset into training and validation subsets in the ratio 2/3 to 1/3
	subset=int32(random.permutation(feats_train.get_num_vectors()))
	vsubset=subset[1:subset.size/3]
	trsubset=subset[1+subset.size/3:subset.size]

	# C4.5 Tree formation using training subset
	train_labels.add_subset(trsubset)
	feats_train.add_subset(trsubset)

	c=C45ClassifierTree()
	c.set_labels(train_labels)
	c.set_feature_types(ft)
	c.train(feats_train)

	train_labels.remove_subset()
	feats_train.remove_subset()

	# prune tree using validation subset
	train_labels.add_subset(vsubset)
	feats_train.add_subset(vsubset)

	c.prune_tree(feats_train,train_labels)

	train_labels.remove_subset()
	feats_train.remove_subset()

	# Classify test data
	output=c.apply_multiclass(feats_test).get_labels()
	output_certainty=c.get_certainty_vector()

	return c,output,output_certainty

if __name__=='__main__':
	print('C45ClassifierTree')
	multiclass_c45classifiertree_modular(*parameter_list[0])
