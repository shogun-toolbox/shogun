#!/usr/bin/env python
import shogun as sg
strings=['hey','guys','i','am','a','string']

parameter_list=[[strings]]

def features_hasheddocdot(strings):
	#create string features
	f=sg.create_string_features(strings, sg.RAWBYTE, sg.PT_CHAR)

	#set the number of bits of the target dimension
	#means a dim of size 2^5=32
	num_bits=5

	#create the ngram tokenizer of size 8 to parse the strings
	tokenizer=sg.NGramTokenizer(8)

	#normalize results
	normalize=True

	#create HashedDocDot features
	hddf = sg.create_features("HashedDocDotFeatures", num_bits=num_bits, 
		 					  doc_collection=f, tokenizer=tokenizer, 
		 					  should_normalize=normalize)
	#should expect 32
	#print('Feature space dimensionality is', hddf.get_dim_feature_space())

	#print('Self dot product of string 0', hddf.dot(0, hddf, 0))

	return hddf

if __name__=='__main__':
	print('HashedDocDotFeatures')
	features_hasheddocdot(*parameter_list[0])
