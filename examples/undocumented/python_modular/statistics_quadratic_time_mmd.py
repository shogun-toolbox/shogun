#!/usr/bin/env python
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *

def statistics_quadratic_time_mmd ():
	from shogun.Features import RealFeatures
	from shogun.Features import MeanShiftRealDataGenerator
	from shogun.Kernel import GaussianKernel
	from shogun.Statistics import QuadraticTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD2_SPECTRUM, MMD2_GAMMA, BIASED, UNBIASED
	from shogun.Distance import EuclideanDistance
	from shogun.Mathematics import Statistics, IntVector

	# note that the quadratic time mmd has to store kernel matrices
	# which upper bounds the sample size
	n=500
	dim=2
	difference=0.5

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftRealDataGenerator(0, dim)
	gen_q=MeanShiftRealDataGenerator(difference, dim)
	
	# Stream examples and merge them in order to compute median on joint sample
	# alternative is to call a different constructor of QuadraticTimeMMD
	features=gen_p.get_streamed_features(n)
	features=features.create_merged_copy(gen_q.get_streamed_features(n))
	
	# use data generator class to produce example data
	data=features.get_feature_matrix()
	
	print "dimension means of X", mean(data.T[0:n].T)
	print "dimension means of Y", mean(data.T[n:2*n+1].T)

	# compute median data distance in order to use for Gaussian kernel width
	# 0.5*median_distance normally (factor two in Gaussian kernel)
	# However, shoguns kernel width is different to usual parametrization
	# Therefore 0.5*2*median_distance^2
	# Use a subset of data for that, only 200 elements. Median is stable
	# Use a permutation set to temporarily merge features in merged examples
	subset=IntVector.randperm_vec(features.get_num_vectors())
	subset=subset[0:200]
	features.add_subset(subset)
	dist=EuclideanDistance(features, features)
	distances=dist.get_distance_matrix()
	features.remove_subset()
	median_distance=Statistics.matrix_median(distances, True)
	sigma=median_distance**2
	print "median distance for Gaussian kernel:", sigma
	kernel=GaussianKernel(10,sigma)

	mmd=QuadraticTimeMMD(kernel,features, n)

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
	null_samples=mmd.sample_null_spectrum(50,10)
	print "null mean:", mean(null_samples)
	print "null variance:", var(null_samples)
	
if __name__=='__main__':
	print('QuadraticTimeMMD')
	statistics_quadratic_time_mmd()
