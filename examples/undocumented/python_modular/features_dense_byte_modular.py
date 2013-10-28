#!/usr/bin/env python
import numpy

# create dense matrix A
A=numpy.array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=numpy.uint8)

parameter_list=[[A]]

def features_dense_byte_modular (A):
	from modshogun import ByteFeatures

	# create dense features a
	# ... of type Byte
	a=ByteFeatures(A)

	# print(some statistics about a)
	#print(a.get_num_vectors())
	#print(a.get_num_features())

	# get first feature vector and set it
	#print(a.get_feature_vector(0))
	a.set_feature_vector(numpy.array([1,4,0,0,0,9], dtype=numpy.uint8), 0)

	# get matrix
	a_out = a.get_feature_matrix()

	#print(type(a_out), a_out.dtype)
	#print(a_out )
	assert(numpy.all(a_out==A))
	return a_out,a

if __name__=='__main__':
	print('ByteFeatures')
	features_dense_byte_modular(*parameter_list[0])
