#!/usr/bin/env python

strings=['example document 1','example document 2','example document 3','example document 4']

parameter_list=[[strings]]

def converter_hasheddoc_modular(strings):
	from modshogun import SparseRealFeatures, RAWBYTE, StringCharFeatures, Features, HashedDocDotFeatures
	from modshogun import NGramTokenizer
	from modshogun import HashedDocConverter
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

	#create converter
	converter=HashedDocConverter(tokenizer, num_bits, normalize)

	converted_feats=converter.apply(f)

	#should expect 32
	print('Converted features\' space dimensionality is', converted_feats.get_dim_feature_space())
	
	print('Self dot product of string 0 with converted feats:', converted_feats.dot(0, converted_feats, 0))

	hashed_feats=HashedDocDotFeatures(num_bits, f, tokenizer, normalize)

	print('Hashed features\' space dimensionality is', hashed_feats.get_dim_feature_space())
	
	print('Self dot product of string 0 with hashed feats:', hashed_feats.dot(0, hashed_feats, 0))

	return converted_feats 

if __name__=='__main__':
	print('HashedDocConverter')
	converter_hasheddoc_modular(*parameter_list[0])


