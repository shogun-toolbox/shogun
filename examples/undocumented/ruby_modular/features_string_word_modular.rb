# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
strings=['hey','guys','string']

parameter_list=[[strings,0,2,0,False]]

def features_string_word_modular(strings, start, order, gap, rev)

	#create string features
# *** 	cf=StringCharFeatures(strings, RAWBYTE)
	cf=Modshogun::StringCharFeatures.new
	cf.set_features(strings, RAWBYTE)
# *** 	wf=StringWordFeatures(RAWBYTE)
	wf=Modshogun::StringWordFeatures.new
	wf.set_features(RAWBYTE)

	wf.obtain_from_char(cf, start, order, gap, rev)

	#and output several stats
	#	puts "max string length", wf.get_max_vector_length()
	#	puts "number of strings", wf.get_num_vectors()
	#	puts "length of first string", wf.get_vector_length(0)
	#	puts "string[2]", wf.get_feature_vector(2)
	#	puts "strings", wf.get_features()

	#replace string 0
	wf.set_feature_vector(array([1,2,3,4,5], dtype=uint16), 0)

	#	puts "strings", wf.get_features()
	return wf.get_features(), wf


end
if __FILE__ == $0
	puts 'StringWordFeatures'
	features_string_word_modular(*parameter_list[0])

end
