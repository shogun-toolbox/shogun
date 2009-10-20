from shogun.Features import RealFeatures, LongIntFeatures, ByteFeatures
from numpy import array, float64, int64, uint8, all

# create dense matrices A,B,C
A=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=float64)
B=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)
C=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=uint8)

# ... of type Real, LongInt and Byte
a=RealFeatures(A)
b=LongIntFeatures(B)
c=ByteFeatures(C)

# or 16bit wide ...
#feat1 = f.ShortFeatures(N.zeros((10,5),N.short))
#feat2 = f.WordFeatures(N.zeros((10,5),N.uint16))


# print some statistics about a
print a.get_num_vectors()
print a.get_num_features()

# get first feature vector and set it
print a.get_feature_vector(0)
a.set_feature_vector(array([1,4,0,0,0,9], dtype=float64), 0)

# get matrices
a_out = a.get_feature_matrix()
b_out = b.get_feature_matrix()
c_out = c.get_feature_matrix()

print type(a_out), a_out.dtype
print a_out 
assert(all(a_out==A))

print type(b_out), b_out.dtype
print b_out 
assert(all(b_out==B))

print type(c_out), c_out.dtype
print c_out 
assert(all(c_out==C))
