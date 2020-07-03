#!/usr/bin/env python
import shogun as sg
import numpy as np

strings=['hey','guys','i','am','a','string']

parameter_list=[[strings]]

def features_string_char (strings):
	#create string features
	f=sg.create_string_features(strings, sg.RAWBYTE, sg.PT_CHAR)

	#and output several stats
	#print("max string length", f.get_max_vector_length())
	#print("number of strings", f.get_num_vectors())
	#print("length of first string", f.get_vector_length(0))
	#print("string[5]", ''.join(f.get_feature_vector(5)))
	#print("strings", f.get_features())

	# FIXME: replace method?
	# replace string 0
	# f.put("string_list", np.array(['t','e','s','t']), 0)

	#print("strings", f.get_features())
	return f.get("string_list"), f

if __name__=='__main__':
	print('StringCharFeatures')
	features_string_char(*parameter_list[0])
