# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
strings=['hey','guys','i','am','a','string']

parameter_list=[[strings]]

def features_string_char_modular(strings)

	#create string features
	f=Modshogun::StringCharFeatures.new(strings, Modshogun::RAWBYTE)

	#and output several stats
	#print "max string length", f.get_max_vector_length()
	#print "number of strings", f.get_num_vectors()
	#print "length of first string", f.get_vector_length(0)
	#print "string[5]", ''.join(f.get_feature_vector(5))
	#print "strings", f.get_features()

	#replace string 0
	f.set_feature_vector(['t','e','s','t'], 0)

	#print "strings", f.get_features()
	return f.get_features(), f


end
if __FILE__ == $0
	print 'StringCharFeatures'
	features_string_char_modular(*parameter_list[0])

end
