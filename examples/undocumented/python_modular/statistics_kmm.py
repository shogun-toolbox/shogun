#!/usr/bin/env python
from numpy import *
from numpy import random

parameter_list = [[10,3]]

def statistics_kmm (n,d):
	from modshogun import RealFeatures
	from modshogun import DataGenerator
	from modshogun import GaussianKernel, MSG_DEBUG
	from modshogun import KernelMeanMatching
	from modshogun import Math

	# init seed for reproducability
	Math.init_random(1)
	random.seed(1);

	data = random.randn(d,n)

	# create shogun feature representation
	features=RealFeatures(data)

	# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
	# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
	# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
	kernel=GaussianKernel(10,8)
	kernel.init(features,features)

	kmm = KernelMeanMatching(kernel,array([0,1,2,3,7,8,9],dtype=int32),array([4,5,6],dtype=int32))
	w = kmm.compute_weights()
	#print w
	return w

if __name__=='__main__':
	print('KernelMeanMatching')
	statistics_kmm(*parameter_list[0])
