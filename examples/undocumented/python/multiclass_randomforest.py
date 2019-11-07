#!/usr/bin/env python
from numpy import array

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False, False])

parameter_list = [[traindat,testdat,label_traindat,feattypes]]

def multiclass_randomforest(train=traindat,test=testdat,labels=label_traindat,ft=feattypes):
	try:
		from shogun import MulticlassLabels, CSVFile, MajorityVote
	except ImportError:
		print("Could not import Shogun modules")
		return
	import shogun as sg

	# wrap features and labels into Shogun objects
	feats_train=sg.features(CSVFile(train))
	feats_test=sg.features(CSVFile(test))
	train_labels=MulticlassLabels(CSVFile(labels))

	# Random Forest formation
	rand_forest=sg.machine("RandomForest", features=feats_train, labels=train_labels,num_bags=20)
	m = rand_forest.get("machine")
	m.put("m_randsubset_size", 1)
	m.put("nominal", ft)

	rand_forest.put("combination_rule", sg.combination_rule("MajorityVote"))
	rand_forest.get_global_parallel().set_num_threads(1)
	rand_forest.train()

	# Classify test data
	output=rand_forest.apply_multiclass(feats_test).get("labels")
	print(output)

	return rand_forest,output

if __name__=='__main__':
	print('RandomForest')
	multiclass_randomforest(*parameter_list[0])
