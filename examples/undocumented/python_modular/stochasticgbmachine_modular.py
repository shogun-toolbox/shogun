#!/usr/bin/env python
import numpy as np

traindat = '../../../data/multiclass/categorical_dataset/fm_housing.dat'
label_traindat = '../../../data/multiclass/categorical_dataset/housing_label.dat'

# set both input attributes as nominal (True) / continuous (False)
feat_types=np.array([False,False,False,True,False,False,False,False,False,False,False,False,False])

parameter_list = [[traindat,label_traindat,feat_types]]

def stochasticgbmachine_modular(train=traindat,train_labels=label_traindat,ft=feat_types):	
	try:
		from modshogun import RealFeatures, RegressionLabels, CSVFile, CARTree, StochasticGBMachine, SquaredLoss
	except ImportError:
		print("Could not import Shogun modules")
		return

	# wrap features and labels into Shogun objects
	feats=RealFeatures(CSVFile(train))
	labels=RegressionLabels(CSVFile(train_labels))

	# divide into training (90%) and test dataset (10%)
	p=np.random.permutation(labels.get_num_labels())
	num=labels.get_num_labels()*0.9

	cart=CARTree()
	cart.set_feature_types(ft)
	cart.set_max_depth(1)
	loss=SquaredLoss()
	s=StochasticGBMachine(cart,loss,500,0.01,0.6)

	# train
	feats.add_subset(np.int32(p[0:num]))
	labels.add_subset(np.int32(p[0:num]))
	s.set_labels(labels)
	s.train(feats)
	feats.remove_subset()
	labels.remove_subset()

	# apply
	feats.add_subset(np.int32(p[num:len(p)]))
	labels.add_subset(np.int32(p[num:len(p)]))
	output=s.apply_regression(feats)

	feats.remove_subset()
	labels.remove_subset()

	return s,output

if __name__=='__main__':
	print('StochasticGBMachine')
	stochasticgbmachine_modular(*parameter_list[0])
