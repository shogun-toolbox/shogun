#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *

def statistics_linear_time_mmd():
	from shogun.Features import RealFeatures
	from shogun.Features import DataGenerator
	from shogun.Kernel import GaussianKernel
	from shogun.Statistics import LinearTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD1_GAUSSIAN

	# note that the linear time statistic is designed for much larger datasets
	n=10000
	dim=2
	difference=0.5

	# use data generator class to produce example data
	# in pratice, this generate data function could be replaced by a method
	# that obtains data from a stream
	data=DataGenerator.generate_mean_data(n,dim,difference)
	
	print "dimension means of X", mean(data.T[0:n].T)
	print "dimension means of Y", mean(data.T[n:2*n+1].T)

	# create shogun feature representation
	features=RealFeatures(data)

	# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
	# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
	# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
	kernel=GaussianKernel(10,8)

	mmd=LinearTimeMMD(kernel,features, n)

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05
	statistic=mmd.compute_statistic()
	print "test statistic:", statistic
	
	# do the same thing using two different way to approximate null-dstribution
	# bootstrapping and gaussian approximation (ony for really large samples)
	alpha=0.05

	print "computing p-value using bootstrapping"
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_bootstrap_iterations(50) # normally, far more iterations are needed
	p_value=mmd.compute_p_value(statistic)
	print "p_value:", p_value
	print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	print "computing p-value using gaussian approximation"
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)
	p_value=mmd.compute_p_value(statistic)
	print "p_value:", p_value
	print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_bootstrap_iterations(10) # normally, far more iterations are needed
	null_samples=mmd.bootstrap_null()
	print "null mean:", mean(null_samples)
	print "null variance:", var(null_samples)
	
if __name__=='__main__':
	print('LinearTimeMMD')
	statistics_linear_time_mmd()
