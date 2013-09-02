#!/usr/bin/env python
strings=['hey','guys','string']

parameter_list=[[strings,0,2,0,False]]

def features_string_word_modular (strings, start, order, gap, rev):
	from modshogun import StringCharFeatures, StringWordFeatures, RAWBYTE
	from numpy import array, uint16

	#create string features
	cf=StringCharFeatures(strings, RAWBYTE)
	wf=StringWordFeatures(RAWBYTE)

	wf.obtain_from_char(cf, start, order, gap, rev)

	#and output several stats
	#print("max string length", wf.get_max_vector_length())
	#print("number of strings", wf.get_num_vectors())
	#print("length of first string", wf.get_vector_length(0))
	#print("string[2]", wf.get_feature_vector(2))
	#print("strings", wf.get_features())

	#replace string 0
	wf.set_feature_vector(array([1,2,3,4,5], dtype=uint16), 0)

	#print("strings", wf.get_features())
	return wf.get_features(), wf

if __name__=='__main__':
	print('StringWordFeatures')
	features_string_word_modular(*parameter_list[0])
