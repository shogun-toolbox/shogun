# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list = [['features_string_file_char_modular.py']]

def features_string_file_char_modular(fname)
	f = StringFileCharFeatures(fname, RAWBYTE)
	#	puts "strings", f.get_features()
	return f


end
if __FILE__ == $0
	puts 'Compressing StringCharFileFeatures'
    features_string_file_char_modular(*parameter_list[0])

end
