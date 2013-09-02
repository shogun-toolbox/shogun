#!/usr/bin/env python
parameter_list = [['features_string_file_char_modular.py']]

def features_string_file_char_modular (fname):
	from modshogun import StringFileCharFeatures, RAWBYTE
	f = StringFileCharFeatures(fname, RAWBYTE)
	#print("strings", f.get_features())
	return f

if __name__=='__main__':
    print('Compressing StringCharFileFeatures')
    features_string_file_char_modular(*parameter_list[0])
