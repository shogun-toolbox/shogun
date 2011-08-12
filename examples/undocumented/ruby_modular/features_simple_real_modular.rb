# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

# create dense matrices A,B,C
matrix=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=float64)

parameter_list = [[matrix]]

# ... of type LongInt
def features_simple_real_modular(A=matrix)
  

end
# ... of type Real, LongInt and Byte
    a=RealFeatures(A)

# print some statistics about a
#print a.get_num_vectors()
#print a.get_num_features()

# get first feature vector and set it
#print a.get_feature_vector(0)
    a.set_feature_vector(array([1,4,0,0,0,9], dtype=float64), 0)

# get matrix
    a_out = a.get_feature_matrix()

    assert(all(a_out==A))
    return a_out


if __FILE__ == $0
    print 'simple_real'
    features_simple_real_modular(*parameter_list[0])

end
