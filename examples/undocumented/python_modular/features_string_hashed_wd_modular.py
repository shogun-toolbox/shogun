#!/usr/bin/env python
from modshogun import LongIntFeatures
from numpy import array, int64, all

# create dense matrix A
matrix=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=int64)

parameter_list = [[matrix,3,1,2],[matrix,3,1,2]]

# ... of type LongInt
def features_string_hashed_wd_modular (A=matrix,order=3,start_order=1,hash_bits=2):
    a=LongIntFeatures(A)

    from numpy import array, uint8
    from modshogun import HashedWDFeatures, StringByteFeatures, RAWDNA
    from modshogun import MSG_DEBUG

    x=[array([0,1,2,3,0,1,2,3,3,2,2,1,1],dtype=uint8)]
    from_order=order
    f=StringByteFeatures(RAWDNA)
    #f.io.set_loglevel(MSG_DEBUG)
    f.set_features(x)

    y=HashedWDFeatures(f,start_order,order,from_order,hash_bits)
    fm=y.get_computed_dot_feature_matrix()

    return fm

if __name__=='__main__':
    print('string_hashed_wd')
    features_string_hashed_wd_modular(*parameter_list[0])
