#!/usr/bin/env python
from shogun import LongIntFeatures
from numpy import array, int64, all

# create dense matrix A
matrix=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)

parameter_list = [[matrix]]

# ... of type LongInt
def features_dense_longint (A=matrix):
	a=LongIntFeatures(A)

	# get matrix
	a_out = a.get_feature_matrix()

	assert(all(a_out==A))
	return a_out

if __name__=='__main__':
	print('dense_longint')
	features_dense_longint(*parameter_list[0])
