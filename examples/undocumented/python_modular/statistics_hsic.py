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
#from pylab import *
from math import pi

parameter_list = [[250,3,3]]

def statistics_hsic (n, difference, angle):
	from modshogun import RealFeatures
	from modshogun import DataGenerator
	from modshogun import GaussianKernel
	from modshogun import HSIC
	from modshogun import BOOTSTRAP, HSIC_GAMMA
	from modshogun import EuclideanDistance
	from modshogun import Math, Statistics, IntVector

	# init seed for reproducability
	Math.init_random(1)

	# note that the HSIC has to store kernel matrices
	# which upper bounds the sample size

	# use data generator class to produce example data
	data=DataGenerator.generate_sym_mix_gauss(n,difference,angle)
	#plot(data[0], data[1], 'x');show()

	# create shogun feature representation
	features_x=RealFeatures(array([data[0]]))
	features_y=RealFeatures(array([data[1]]))

	# compute median data distance in order to use for Gaussian kernel width
	# 0.5*median_distance normally (factor two in Gaussian kernel)
	# However, shoguns kernel width is different to usual parametrization
	# Therefore 0.5*2*median_distance^2
	# Use a subset of data for that, only 200 elements. Median is stable
	subset=IntVector.randperm_vec(features_x.get_num_vectors())
	subset=subset[0:200]
	features_x.add_subset(subset)
	dist=EuclideanDistance(features_x, features_x)
	distances=dist.get_distance_matrix()
	features_x.remove_subset()
	median_distance=Statistics.matrix_median(distances, True)
	sigma_x=median_distance**2
	features_y.add_subset(subset)
	dist=EuclideanDistance(features_y, features_y)
	distances=dist.get_distance_matrix()
	features_y.remove_subset()
	median_distance=Statistics.matrix_median(distances, True)
	sigma_y=median_distance**2
	#print "median distance for Gaussian kernel on x:", sigma_x
	#print "median distance for Gaussian kernel on y:", sigma_y
	kernel_x=GaussianKernel(10,sigma_x)
	kernel_y=GaussianKernel(10,sigma_y)

	hsic=HSIC(kernel_x,kernel_y,features_x,features_y)

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05 using different methods to approximate
	# null-distribution
	statistic=hsic.compute_statistic()
	#print "HSIC:", statistic
	alpha=0.05

	#print "computing p-value using bootstrapping"
	hsic.set_null_approximation_method(BOOTSTRAP)
	# normally, at least 250 iterations should be done, but that takes long
	hsic.set_bootstrap_iterations(100)
	# bootstrapping allows usage of unbiased or biased statistic
	p_value_boot=hsic.compute_p_value(statistic)
	thresh_boot=hsic.compute_threshold(alpha)
	#print "p_value:", p_value_boot
	#print "threshold for 0.05 alpha:", thresh_boot
	#print "p_value <", alpha, ", i.e. test sais p and q are dependend:", p_value_boot<alpha

	#print "computing p-value using gamma method"
	hsic.set_null_approximation_method(HSIC_GAMMA)
	p_value_gamma=hsic.compute_p_value(statistic)
	thresh_gamma=hsic.compute_threshold(alpha)
	#print "p_value:", p_value_gamma
	#print "threshold for 0.05 alpha:", thresh_gamma
	#print "p_value <", alpha, ", i.e. test sais p and q are dependend::", p_value_gamma<alpha

	# sample from null distribution (these may be plotted or whatsoever)
	# mean should be close to zero, variance stronly depends on data/kernel
	# bootstrapping, biased statistic
	#print "sampling null distribution using bootstrapping"
	hsic.set_null_approximation_method(BOOTSTRAP)
	hsic.set_bootstrap_iterations(100)
	null_samples=hsic.bootstrap_null()
	#print "null mean:", mean(null_samples)
	#print "null variance:", var(null_samples)
	#hist(null_samples, 100); show()

	return p_value_boot, thresh_boot, p_value_gamma, thresh_gamma, statistic, null_samples

if __name__=='__main__':
	print('HSIC')
	statistics_hsic(*parameter_list[0])
