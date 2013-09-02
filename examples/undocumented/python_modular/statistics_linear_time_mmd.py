#!/usr/bin/env python
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012-2013 Heiko Strathmann
#
from numpy import *

parameter_list = [[1000,2,0.5]]

def statistics_linear_time_mmd (n,dim,difference):
	from modshogun import RealFeatures
	from modshogun import MeanShiftDataGenerator
	from modshogun import GaussianKernel
	from modshogun import LinearTimeMMD
	from modshogun import BOOTSTRAP, MMD1_GAUSSIAN
	from modshogun import EuclideanDistance
	from modshogun import Statistics, Math

	# init seed for reproducability
	Math.init_random(1)

	# note that the linear time statistic is designed for much larger datasets
	# so increase to get reasonable results

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftDataGenerator(0, dim)
	gen_q=MeanShiftDataGenerator(difference, dim)

	# compute median data distance in order to use for Gaussian kernel width
	# 0.5*median_distance normally (factor two in Gaussian kernel)
	# However, shoguns kernel width is different to usual parametrization
	# Therefore 0.5*2*median_distance^2
	# Use a subset of data for that, only 200 elements. Median is stable
	
	# Stream examples and merge them in order to compute median on joint sample
	features=gen_p.get_streamed_features(100)
	features=features.create_merged_copy(gen_q.get_streamed_features(100))
	
	# compute all pairwise distances
	dist=EuclideanDistance(features, features)
	distances=dist.get_distance_matrix()
	
	# compute median and determine kernel width (using shogun)
	median_distance=Statistics.matrix_median(distances, True)
	sigma=median_distance**2
	#print "median distance for Gaussian kernel:", sigma
	kernel=GaussianKernel(10,sigma)

	# mmd instance using streaming features, blocksize of 10000
	mmd=LinearTimeMMD(kernel, gen_p, gen_q, n, 10000)

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05
	statistic=mmd.compute_statistic()
	#print "test statistic:", statistic
	
	# do the same thing using two different way to approximate null-dstribution
	# bootstrapping and gaussian approximation (ony for really large samples)
	alpha=0.05

	#print "computing p-value using bootstrapping"
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_bootstrap_iterations(50) # normally, far more iterations are needed
	p_value_boot=mmd.compute_p_value(statistic)
	#print "p_value_boot:", p_value_boot
	#print "p_value_boot <", alpha, ", i.e. test sais p!=q:", p_value_boot<alpha
	
	#print "computing p-value using gaussian approximation"
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)
	p_value_gaussian=mmd.compute_p_value(statistic)
	#print "p_value_gaussian:", p_value_gaussian
	#print "p_value_gaussian <", alpha, ", i.e. test sais p!=q:", p_value_gaussian<alpha
	
	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	mmd.set_null_approximation_method(BOOTSTRAP)
	mmd.set_bootstrap_iterations(10) # normally, far more iterations are needed
	null_samples=mmd.bootstrap_null()
	#print "null mean:", mean(null_samples)
	#print "null variance:", var(null_samples)
	
	# compute type I and type II errors for Gaussian approximation
	# number of trials should be larger to compute tight confidence bounds
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)
	num_trials=5;
	alpha=0.05 # test power
	typeIerrors=[0 for x in range(num_trials)]
	typeIIerrors=[0 for x in range(num_trials)]
	for i in range(num_trials):
		# this effectively means that p=q - rejecting is tpye I error
		mmd.set_simulate_h0(True)
		typeIerrors[i]=mmd.perform_test()>alpha
		mmd.set_simulate_h0(False)
		
		typeIIerrors[i]=mmd.perform_test()>alpha
	
	#print "type I error:", mean(typeIerrors), ", type II error:", mean(typeIIerrors)
	
	return statistic, p_value_boot, p_value_gaussian, null_samples, typeIerrors, typeIIerrors
	
if __name__=='__main__':
	print('LinearTimeMMD')
	statistics_linear_time_mmd(*parameter_list[0])
