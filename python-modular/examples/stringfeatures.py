import shogun.Features as sf
import shogun.Library as sl

f=sf.StringCharFeatures(sf.Alphabet(sf.RAWBYTE))
print f.load_from_directory(".")
print f.get_max_vector_length()
print f.get_num_vectors()
print f.get_vector_length(0)
print f.get_feature(0,0)
print f.get_feature(0,1)
print f.get_feature(0,2)
f.select_feature_vector(0)
v=f.get_str()
print v

f2=sf.StringCharFeatures(sf.Alphabet(sf.DNA))
f2.set_string_features(['hey','guys','i','am','a','string'])

f2.select_feature_vector(1)
v=f2.get_str()
print v

s=10*'a' + 10*'b' + 10*'c'
f3=sf.StringCharFeatures(sf.Alphabet(sf.RAWBYTE))
f3.set_string_features([s])
f3.obtain_by_sliding_window(5,1)
print f3.get_num_vectors()
print f3.get_vector_length(0)
print f3.get_vector_length(1)

print s
for i in xrange(f3.get_num_vectors()):
	f3.select_feature_vector(i)
	v=f3.get_str()
	print `i`+ ':',
	print v

f4=sf.StringCharFeatures(sf.Alphabet(sf.RAWBYTE))
f4.set_string_features([s])
positions=sl.DynamicIntArray()
positions.append_element(0)
positions.append_element(6)
positions.append_element(16)
positions.append_element(25)
#positions.append_element(28)
f4.obtain_by_position_list(5,positions)

print s
for i in xrange(f4.get_num_vectors()):
	f4.select_feature_vector(i)
	v=f4.get_str()
	print `i`+ ':',
	print v
