#!/usr/bin/env python

from numpy import *

parameter_list = [[100, 2, 5,1.,1000,1,1], [100, 2, 5,1.,1000,1,2]]

def classifier_perceptron_modular (n=100, dim=2, distance=5,learn_rate=1.,max_iter=1000,num_threads=1,seed=1):
	from modshogun import RealFeatures, BinaryLabels
	from modshogun import Perceptron

	random.seed(seed)

	# produce some (probably) linearly separable training data by hand
	# Two Gaussians at a far enough distance
	X=array(random.randn(dim,n))+distance
	Y=array(random.randn(dim,n))-distance
	X_test=array(random.randn(dim,n))+distance
	Y_test=array(random.randn(dim,n))-distance
	label_train_twoclass=hstack((ones(n), -ones(n)))

	#plot(X[0,:], X[1,:], 'x', Y[0,:], Y[1,:], 'o')
	fm_train_real=hstack((X,Y))
	fm_test_real=hstack((X_test,Y_test))

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=BinaryLabels(label_train_twoclass)

	perceptron=Perceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	# only guaranteed to converge for separable data
	perceptron.train()

	perceptron.set_features(feats_test)
	out_labels = perceptron.apply().get_labels()
	return perceptron, out_labels

if __name__=='__main__':
	print('Perceptron')
	classifier_perceptron_modular(*parameter_list[0])
