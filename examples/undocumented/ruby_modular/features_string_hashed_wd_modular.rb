# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

# create dense matrix A
matrix=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)

parameter_list = [[matrix,3,1,2],[matrix,3,1,2]]

# ... of type LongInt
def features_string_hashed_wd_modular(A=matrix,order=3,start_order=1,hash_bits=2)
    a=LongIntFeatures(A)
    

    x=[array([0,1,2,3,0,1,2,3,3,2,2,1,1],dtype=uint8)]
    from_order=order
    f=StringByteFeatures(RAWDNA)
    #f.io.set_loglevel(MSG_DEBUG)
    f.set_features(x)

    y=HashedWDFeatures(f,start_order,order,from_order,hash_bits)
    fm=y.get_computed_dot_feature_matrix()

    return fm


end
if __FILE__ == $0
    print 'string_hashed_wd'
    features_string_hashed_wd_modular(*parameter_list[0])

end
