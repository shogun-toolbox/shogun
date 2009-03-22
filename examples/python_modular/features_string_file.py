from shogun.Features import StringCharFeatures, RAWBYTE

# load features from file
f=StringCharFeatures(RAWBYTE)
f.load_from_directory(".")

#and output several stats
print "max string length", f.get_max_vector_length()
print "number of strings", f.get_num_vectors()
print "length of first string", f.get_vector_length(0)
print "str[0,0:3]", f.get_feature(0,0), f.get_feature(0,1), f.get_feature(0,2)
print "len(str[0])", f.get_vector_length(0)
print "str[0]", f.get_feature_vector(0)

