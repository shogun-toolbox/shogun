#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2014 Wu Lin
# Copyright (C) 2014 Wu Lin

path='../data'
traindat = '%s/fm_train_real.dat'%path
testdat = '%s/fm_test_real.dat'%path
label_traindat = '%s/label_train_twoclass.dat'%path

from modshogun import *
parameter_list=[[KLCholeskyInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],[KLFullDiagonalInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],[KLApproxDiagonalInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],[KLDualInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0]]


def variational_classifier_modular(kl_inference,train_fname=traindat,test_fname=testdat,label_fname=label_traindat,kernel_log_sigma=0,kernel_log_scale=0,noise_factor=1e-5,min_coeff_kernel=1e-2,max_attempt=0):
	from math import exp
	features_train=RealFeatures(CSVFile(train_fname))
	labels_train=BinaryLabels(CSVFile(label_fname))

	likelihood=LogitDVGLikelihood()
	error_eval=ContingencyTableEvaluation()
	mean_func=ConstMean()
	kernel_sigma=2*exp(2*kernel_log_sigma);
	kernel_func=GaussianKernel(10, kernel_sigma)

	inf=kl_inference(kernel_func, features_train, mean_func, labels_train, likelihood)
	try:
		inf.set_noise_factor(noise_factor)
		inf.set_min_coeff_kernel(min_coeff_kernel)
		inf.set_max_attempt(max_attempt)
	except:
		pass
	inf.set_scale(exp(kernel_log_scale))
	gp=GaussianProcessBinaryClassification(inf)
	gp.train()
	pred_labels_train=gp.apply_binary(features_train)
	error_train=error_eval.evaluate(pred_labels_train, labels_train)
	print "\nInference name:%s"%inf.get_name(),
	print "marginal likelihood:%.10f"%inf.get_negative_log_marginal_likelihood(),
	print "Training error %.4f"%error_train
	return pred_labels_train, gp, pred_labels_train.get_labels()


if __name__=="__main__":
	print("variational_classifier")
	for parameter in parameter_list: 
		variational_classifier_modular(*parameter)
