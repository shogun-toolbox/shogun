#!/usr/bin/env python
import numpy
# create dense matrix A
A=numpy.array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=numpy.float64)

parameter_list=[[A]]
def features_sparse_modular (A):
	from scipy.sparse import csc_matrix
	from modshogun import SparseRealFeatures
	from numpy import array, float64, all

	# sparse representation X of dense matrix A
	# note, will work with types other than float64 too,
	# but requires recent scipy.sparse
	X=csc_matrix(A)
	#print(A)

	# create sparse shogun features from dense matrix A
	a=SparseRealFeatures(A)
	a_out=a.get_full_feature_matrix()
	#print(a_out)
	assert(all(a_out==A))
	#print(a_out)

	# create sparse shogun features from sparse matrix X
	a.set_sparse_feature_matrix(X)
	a_out=a.get_full_feature_matrix()
	#print(a_out)
	assert(all(a_out==A))

	# create sparse shogun features from sparse matrix X
	a=SparseRealFeatures(X)
	a_out=a.get_full_feature_matrix()
	#print(a_out)
	assert(all(a_out==A))

	# obtain (data,row,indptr) csc arrays of sparse shogun features
	z=csc_matrix(a.get_sparse_feature_matrix())
	z_out=z.todense()
	#print(z_out)
	assert(all(z_out==A))

if __name__=='__main__':
	print('Sparse Features')
	features_sparse_modular(*parameter_list[0])
