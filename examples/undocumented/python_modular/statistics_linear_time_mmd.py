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

def statistics_linear_time_mmd ():
	from shogun.Features import RealFeatures
	from shogun.Features import MeanShiftRealDataGenerator
	from shogun.Kernel import GaussianKernel
	from shogun.Statistics import LinearTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD1_GAUSSIAN
	from shogun.Distance import EuclideanDistance
	from shogun.Mathematics import Statistics, Math

	# note that the linear time statistic is designed for much larger datasets
	n=10000
	dim=2
	difference=0.5

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftRealDataGenerator(0, dim)
	gen_q=MeanShiftRealDataGenerator(difference, dim)

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
	print "median distance for Gaussian kernel:", sigma
	kernel=GaussianKernel(10,sigma)

	# mmd instance using streaming features, blocksize of 10000
	mmd=LinearTimeMMD(kernel, gen_p, gen_q, n, 10000)

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
