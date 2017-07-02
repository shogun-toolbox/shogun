#!/usr/bin/env python
parameter_list = [['features_string_file_char.py']]

def features_string_file_char (fname):
	from modshogun import StringFileCharFeatures, RAWBYTE
	f = StringFileCharFeatures(fname, RAWBYTE)
	#print("strings", f.get_features())
	return f

if __name__=='__main__':
    print('Compressing StringCharFileFeatures')
    features_string_file_char(*parameter_list[0])
