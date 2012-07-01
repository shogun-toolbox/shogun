#!/usr/bin/env python

from numpy import *

# class for creating pairs of sample sets from two different distribution
class TwoDistributionsData:
	def __init__(self):
		pass

	# creates to sample sets of desired shape which is standard normal distrbuted
	# the first dimension of Y has a mean shift of difference
	def create_mean_data(self, n, dim, difference):
		X=random.randn(n,dim)
		Y=random.randn(n,dim)

		for i in range(len(Y)):
			Y[i][0]+=difference
		
		# return in shogun format: vectors are columns
		return (transpose(X),transpose(Y))
