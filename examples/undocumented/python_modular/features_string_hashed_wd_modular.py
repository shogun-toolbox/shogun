from numpy import *
from shogun.Features import *
from shogun.Library import MSG_DEBUG

order=3
start_order=1
from_order=order
hash_bits=2

x=[array([0,1,2,3,0,1,2,3,3,2,2,1,1],dtype=uint8)]
print len(x[0])

f=StringByteFeatures(RAWDNA)
f.io.set_loglevel(MSG_DEBUG)
f.set_features(x)

y=HashedWDFeatures(f,start_order,order,from_order,hash_bits)
print y.get_dim_feature_space()
fm=y.get_feature_matrix()
print fm.shape
print fm

