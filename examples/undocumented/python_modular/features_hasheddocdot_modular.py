#!/usr/bin/env python
strings=['hey','guys','i','am','a','string']

parameter_list=[[strings]]

def features_hasheddocdot_modular(strings):
	from modshogun import StringCharFeatures, RAWBYTE
	from modshogun import HashedDocDotFeatures
	from modshogun import NGramTokenizer
	from numpy import array

	#create string features
	f=StringCharFeatures(strings, RAWBYTE)

	#set the number of bits of the target dimension
	#means a dim of size 2^5=32
	num_bits=5

	#create the ngram tokenizer of size 8 to parse the strings
	tokenizer=NGramTokenizer(8)

	#normalize results
	normalize=True

	#create HashedDocDot features
	hddf=HashedDocDotFeatures(num_bits, f, tokenizer, normalize)

	#should expect 32
	#print('Feature space dimensionality is', hddf.get_dim_feature_space())

	#print('Self dot product of string 0', hddf.dot(0, hddf, 0))

	return hddf

if __name__=='__main__':
	print('HashedDocDotFeatures')
	features_hasheddocdot_modular(*parameter_list[0])
