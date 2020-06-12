#!/usr/bin/env python
import shogun as sg

strings=['example document 1','example document 2','example document 3','example document 4']

parameter_list=[[strings]]

def converter_hasheddoc(strings):
	#create string features
	f=sg.create_string_features(strings, sg.RAWBYTE, sg.PT_CHAR)

	#set the number of bits of the target dimension
	#means a dim of size 2^5=32
	num_bits=5

	#create the ngram tokenizer of size 8 to parse the strings
	tokenizer=sg.NGramTokenizer(8)

	#normalize results
	normalize=True

	#create converter
	converter = sg.create_transformer('HashedDocConverter', tokenizer=tokenizer, num_bits=num_bits, should_normalize=normalize)

	converted_feats=converter.transform(f)

	#should expect 32
	#print('Converted features\' space dimensionality is', converted_feats.get_dim_feature_space())

	#print('Self dot product of string 0 with converted feats:', converted_feats.dot(0, converted_feats, 0))

	hashed_feats=sg.create_features("HashedDocDotFeatures", num_bits=num_bits, 
									doc_collection=f, tokenizer=tokenizer, 
									should_normalize=normalize)

	#print('Hashed features\' space dimensionality is', hashed_feats.get_dim_feature_space())

	#print('Self dot product of string 0 with hashed feats:', hashed_feats.dot(0, hashed_feats, 0))

	return converted_feats

if __name__=='__main__':
	print('HashedDocConverter')
	converter_hasheddoc(*parameter_list[0])


