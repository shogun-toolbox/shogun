from shogun.Features import StringCharFeatures, StringUlongFeatures, RAWBYTE
from numpy import array, uint64

#create string features
cf=StringCharFeatures(['hey','guys','string'], RAWBYTE)
uf=StringUlongFeatures(RAWBYTE)

#start=0, order=2, gap=0, rev=False)
uf.obtain_from_char(cf, 0, 2, 0, False)

#and output several stats
print "max string length", uf.get_max_vector_length()
print "number of strings", uf.get_num_vectors()
print "length of first string", uf.get_vector_length(0)
print "string[2]", uf.get_feature_vector(2)
print "strings", uf.get_features()

#replace string 0
uf.set_feature_vector(array([1,2,3,4,5], dtype=uint64), 0)

print "strings", uf.get_features()
