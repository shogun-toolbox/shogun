require 'modshogun'
require 'pp'

strings=['hey','guys','i','am','a','string']

parameter_list=[strings]

def features_string_char_modular(strings)

	#create string features
	f=Modshogun::StringCharFeatures.new(strings, Modshogun::RAWBYTE)

	#and output several stats
	#puts "max string length", f.get_max_vector_length
	#puts "number of strings", f.get_num_vectors
	#puts "length of first string", f.get_vector_length(0)
	#puts "string[5]", f.get_feature_vector(5)
	#puts "strings", f.get_features

	#replace string 0
	f.set_feature_vector(['t','e','s','t'], 0)

	return f.get_features, f

end

if __FILE__ == $0
	puts 'StringCharFeatures'
	pp features_string_char_modular(*parameter_list)
end
