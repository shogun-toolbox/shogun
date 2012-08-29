#!/usr/bin/env python
from numpy import *
from numpy import random

def statistics_kmm ():
	from shogun.Features import RealFeatures
	from shogun.Features import DataGenerator
	from shogun.Kernel import GaussianKernel, MSG_DEBUG
	from shogun.Statistics import KernelMeanMatching

	data = random.randn(3,10)

	# create shogun feature representation
	features=RealFeatures(data)

	# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
	# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
	# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
	kernel=GaussianKernel(10,8)
	kernel.init(features,features)

	kmm = KernelMeanMatching(kernel,array([0,1,2,3,7,8,9],dtype=int32),array([4,5,6],dtype=int32))
	w = kmm.compute_weights()
	print w

if __name__=='__main__':
	print('KernelMeanMatching')
	statistics_kmm()
