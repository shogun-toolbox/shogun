from shogun.Features import ByteFeatures
from numpy import array, uint8, all

# create dense matrix A
A=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=uint8)

# ... of type Byte
a=ByteFeatures(A)


# print some statistics about a
print a.get_num_vectors()
print a.get_num_features()

# get first feature vector and set it
print a.get_feature_vector(0)
a.set_feature_vector(array([1,4,0,0,0,9], dtype=uint8), 0)

# get matrix
a_out = a.get_feature_matrix()

print type(a_out), a_out.dtype
print a_out 
assert(all(a_out==A))
