# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list=[[".", "features_string_char_modular.py"]]

def features_string_file_modular(directory, fname)

	# load features from directory
# *** 	f=StringCharFeatures(RAWBYTE)
	f=Modshogun::StringCharFeatures.new
	f.set_features(RAWBYTE)
	f.load_from_directory(directory)

	#and output several stats
	#	puts "max string length", f.get_max_vector_length()
	#	puts "number of strings", f.get_num_vectors()
	#	puts "length of first string", f.get_vector_length(0)
	#	puts "str[0,0:3]", f.get_feature(0,0), f.get_feature(0,1), f.get_feature(0,2)
	#	puts "len(str[0])", f.get_vector_length(0)
	#	puts "str[0]", f.get_feature_vector(0)

	#or load features from file (one string per line)
# *** 	fil=AsciiFile(fname)
	fil=Modshogun::AsciiFile.new
	fil.set_features(fname)
	f.load(fil)
	#	puts f.get_features()

	#or load fasta file
	#f.load_fasta('fasta.fa')
	#	puts f.get_features()
	return f.get_features(), f


end
if __FILE__ == $0
	puts 'StringWordFeatures'
	features_string_file_modular(*parameter_list[0])

end
