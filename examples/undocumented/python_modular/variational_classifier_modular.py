#!/usr/bin/env python

#
# Copyright (c) The Shogun Machine Learning Toolbox
# Written (w) 2014 Wu Lin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the Shogun Development Team.
#
#

path='../data'
traindat = '%s/fm_train_real.dat'%path
testdat = '%s/fm_test_real.dat'%path
label_traindat = '%s/label_train_twoclass.dat'%path

from modshogun import *
parameter_list=[
	[KLCholeskyInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],
	[KLFullDiagonalInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],
	[KLApproxDiagonalInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0],
	[KLDualInferenceMethod,traindat,testdat,label_traindat,0,0,1e-5,1e-2,0]
]


def variational_classifier_modular(kl_inference,train_fname=traindat,test_fname=testdat,
	label_fname=label_traindat,kernel_log_sigma=0,kernel_log_scale=0,noise_factor=1e-5,
	min_coeff_kernel=1e-2,max_attempt=0):
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
	#print "\nInference name:%s"%inf.get_name(),
	#print "marginal likelihood:%.10f"%inf.get_negative_log_marginal_likelihood(),
	#print "Training error %.4f"%error_train
	return pred_labels_train, gp, pred_labels_train.get_labels()


if __name__=="__main__":
	print("variational_classifier")
	for parameter in parameter_list: 
		variational_classifier_modular(*parameter)
