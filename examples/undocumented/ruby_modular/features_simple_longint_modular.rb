# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

# create dense matrix A
matrix=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)

parameter_list = [[matrix]]

# ... of type LongInt
def features_simple_longint_modular(A=matrix)
    a=LongIntFeatures(A)

end
# get first feature vector and set it

    a.set_feature_vector(array([1,4,0,0,0,9], dtype=int64), 0)

# get matrix
    a_out = a.get_feature_matrix()

    
    assert(all(a_out==A))
    return a_out

if __FILE__ == $0
    print 'simple_longint'
    features_simple_longint_modular(*parameter_list[0])

end
