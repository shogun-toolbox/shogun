#!/usr/bin/env python
strings=['hey','guys','i','am','a','string']

parameter_list=[[strings]]

def features_string_char_modular (strings):
	from modshogun import StringCharFeatures, RAWBYTE
	from numpy import array

	#create string features
	f=StringCharFeatures(strings, RAWBYTE)

	#and output several stats
	#print("max string length", f.get_max_vector_length())
	#print("number of strings", f.get_num_vectors())
	#print("length of first string", f.get_vector_length(0))
	#print("string[5]", ''.join(f.get_feature_vector(5)))
	#print("strings", f.get_features())

	#replace string 0
	f.set_feature_vector(array(['t','e','s','t']), 0)

	#print("strings", f.get_features())
	return f.get_features(), f

if __name__=='__main__':
	print('StringCharFeatures')
	features_string_char_modular(*parameter_list[0])
