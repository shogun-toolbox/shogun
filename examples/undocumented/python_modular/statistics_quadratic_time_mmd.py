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
	from shogun.Statistics import QuadraticTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD2_SPECTRUM, MMD2_GAMMA, BIASED, UNBIASED

	# note that the quadratic time mmd has sometimes to store kernel matrices
	# which upper bounds the sample size massively
	n=500
	dim=2
	difference=0.5

	# data is standard normal distributed. only one dimension of Y has a mean
	# shift of difference
	(X,Y)=gen_data.create_mean_data(n,dim,difference)
	
	print "dimension means of X", [mean(x) for x in X]
	print "dimension means of Y", [mean(x) for x in Y]

	# create shogun feature representation
	features_x=RealFeatures(X)
	features_y=RealFeatures(Y)

	# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
	# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
	# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
	kernel=GaussianKernel(10,8)

	mmd=QuadraticTimeMMD(kernel,features_x, features_y)

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05 using different methods to approximate
	# null-distribution
	statistic=mmd.compute_statistic()
	alpha=0.05
	
	print "computing p-value using bootstrapping"
	mmd.set_null_approximation_method(BOOTSTRAP)
	# normally, at least 250 iterations should be done, but that takes long
	mmd.set_bootstrap_iterations(10)
	# bootstrapping allows usage of unbiased or biased statistic
	mmd.set_statistic_type(UNBIASED)
	p_value=mmd.compute_p_value(statistic)
	print "p_value:", p_value
	print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	# only can do this if SHOGUN was compiled with LAPACK so check
	if "sample_null_spectrum" in dir(QuadraticTimeMMD):
		print "computing p-value using spectrum method"
		mmd.set_null_approximation_method(MMD2_SPECTRUM)
		# normally, at least 250 iterations should be done, but that takes long
		mmd.set_num_samples_sepctrum(50)
		mmd.set_num_eigenvalues_spectrum(n-10)
		# spectrum method computes p-value for biased statistics only
		mmd.set_statistic_type(BIASED)
		p_value=mmd.compute_p_value(statistic)
		print "p_value:", p_value
		print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	print "computing p-value using gamma method"
	mmd.set_null_approximation_method(MMD2_GAMMA)
	# gamma method computes p-value for biased statistics only
	mmd.set_statistic_type(BIASED)
	p_value=mmd.compute_p_value(statistic)
	print "p_value:", p_value
	print "p_value <", alpha, ", i.e. test sais p!=q:", p_value<alpha
	
	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	# bootstrapping, biased statistic
	print "sampling null distribution using bootstrapping"
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_statistic_type(BIASED)
	mmd.set_bootstrap_iterations(10)
	null_samples=mmd.bootstrap_null()
	print "null mean:", mean(null_samples)
	print "null variance:", var(null_samples)
	
	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	# spectrum, biased statistic
	print "sampling null distribution using spectrum method"
	mmd.set_null_approximation_method(MMD2_SPECTRUM)
	mmd.set_statistic_type(BIASED)
	# 200 samples using 100 eigenvalues
	null_samples=mmd.sample_null_spectrum(200,100)
	print "null mean:", mean(null_samples)
	print "null variance:", var(null_samples)
	
if __name__=='__main__':
	print('LinearTimeMMD')
	statistics_linear_time_mmd()
