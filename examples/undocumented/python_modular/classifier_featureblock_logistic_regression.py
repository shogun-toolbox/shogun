#!/usr/bin/env python
from numpy import array,hstack
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def classifier_featureblock_logistic_regression (fm_train=traindat,fm_test=testdat,label_train=label_traindat):

	from modshogun import BinaryLabels, RealFeatures, IndexBlock, IndexBlockGroup, FeatureBlockLogisticRegression

	features = RealFeatures(hstack((traindat,traindat)))
	labels = BinaryLabels(hstack((label_train,label_train)))

	n_features = features.get_num_features()
	block_one = IndexBlock(0,n_features//2)
	block_two = IndexBlock(n_features//2,n_features)
	block_group = IndexBlockGroup()
	block_group.add_block(block_one)
	block_group.add_block(block_two)

	mtlr = FeatureBlockLogisticRegression(0.1,features,labels,block_group)
	mtlr.set_regularization(1) # use regularization ratio
	mtlr.set_tolerance(1e-2) # use 1e-2 tolerance
	mtlr.train()
	out = mtlr.apply().get_labels()

	return out

if __name__=='__main__':
	print('FeatureBlockLogisticRegression')
	classifier_featureblock_logistic_regression(*parameter_list[0])
