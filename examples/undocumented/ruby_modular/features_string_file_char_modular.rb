# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
parameter_list = [['features_string_file_char_modular.py']]

def features_string_file_char_modular(fname)
	f = StringFileCharFeatures(fname, RAWBYTE)
	#print "strings", f.get_features()
	return f


end
if __FILE__ == $0
    print 'Compressing StringCharFileFeatures'
    features_string_file_char_modular(*parameter_list[0])

end
