#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *
from tools.two_distributions_data import TwoDistributionsData

gen_data=TwoDistributionsData()

def statistics_linear_time_mmd():
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Statistics import LinearTimeMMD, QuadraticTimeMMD
	from shogun.Statistics import BOOTSTRAP

	import matplotlib.pyplot as plt

	# note that the linear time statistic is designed for much larger datasets
	n=10000
	dim=2
	difference=0.5

	# data is standard normal distributed. only one dimension of Y has a mean
	# shift of difference
	# in pratice, this generate data function could be replaced by a method
	# that obtains data from a stream
	(X,Y)=gen_data.create_mean_data(n,dim,difference)

	print "dimension means of X", [mean(x) for x in X]
	print "dimension means of Y", [mean(x) for x in Y]

	# create shogun feature representation
	features_x=RealFeatures(X)
	features_y=RealFeatures(Y)

	# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
	# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
	# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
	kernel=GaussianKernel(10,0.125)

	mmd=LinearTimeMMD(kernel,features_x, features_y)

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05
	# for the linear time mmd, the statistic has to be computed on different
	# data than the p-value, so first, compute statistic, and then compute
	# p-value on other data
	# this annoying property is since the null-distribution should stay normal
	# which is not the case if "training/test" data would be the same
	statistic=mmd.compute_statistic()
	
	# generate new data (same distributions as old) and new statistic object
	(X,Y)=gen_data.create_mean_data(n,dim,difference)
	features_x=RealFeatures(X)
	features_y=RealFeatures(Y)
	mmd=LinearTimeMMD(kernel,features_x, features_y)
	
	p_value=mmd.compute_p_value(statistic)
	alpha=0.05
	print "p_value:", p_value
	print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_bootstrap_iterations(100)
	null_samples=mmd.bootstrap_null()
	print "null mean:", mean(null_samples)
	print "null variance:", var(null_samples)
	
if __name__=='__main__':
	print('LinearTimeMMD')
	statistics_linear_time_mmd()
