#!/usr/bin/env python
import numpy as np
import shogun as sg

traindat = '../../../data/uci/housing/fm_housing.dat'
label_traindat = '../../../data/uci/housing/housing_label.dat'

# set both input attributes as nominal (True) / continuous (False)
feat_types=np.array([False,False,False,True,False,False,False,False,False,False,False,False,False])

parameter_list = [[traindat,label_traindat,feat_types]]

def stochasticgbmachine(train=traindat,train_labels=label_traindat,ft=feat_types):

	# wrap features and labels into Shogun objects
	feats=sg.create_features(sg.read_csv(train))
	labels=sg.create_labels(sg.read_csv(train_labels))

	# divide into training (90%) and test dataset (10%)
	p=np.random.permutation(labels.get_num_labels())
	num=labels.get_num_labels()*0.9

	cart=sg.create_machine("CARTree", nominal=ft, max_depth=1)
	loss = sg.create_loss('SquaredLoss')
	s=sg.create_machine("StochasticGBMachine", machine=cart, loss=loss, 
		num_iterations=500, learning_rate=0.01)

	# train
	feats.add_subset(np.int32(p[0:int(num)]))
	labels.add_subset(np.int32(p[0:int(num)]))
	s.train(feats, labels)
	feats.remove_subset()
	labels.remove_subset()

	# apply
	feats.add_subset(np.int32(p[int(num):len(p)]))
	labels.add_subset(np.int32(p[int(num):len(p)]))
	output=s.apply_regression(feats)

	feats.remove_subset()
	labels.remove_subset()

	return s,output

if __name__=='__main__':
	print('StochasticGBMachine')
	stochasticgbmachine(*parameter_list[0])
