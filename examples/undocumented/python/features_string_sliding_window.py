#!/usr/bin/env python
# create string features with a single string
s=10*'A' + 10*'C' + 10*'G' + 10*'T'

parameter_list=[[s]]

def features_string_sliding_window_modular (strings):
	from modshogun import StringCharFeatures, DNA
	from modshogun import DynamicIntArray

	f=StringCharFeatures([strings], DNA)

	# slide a window of length 5 over features
	# (memory efficient, does not copy strings)
	f.obtain_by_sliding_window(5,1)
	#print(f.get_num_vectors())
	#print(f.get_vector_length(0))
	#print(f.get_vector_length(1))
	#print(f.get_features())

	# slide a window of length 4 over features
	# (memory efficient, does not copy strings)
	f.obtain_by_sliding_window(4,1)
	#print(f.get_num_vectors())
	#print(f.get_vector_length(0))
	#print(f.get_vector_length(1))
	#print(f.get_features())

	# extract string-windows at position 0,6,16,25 of window size 4
	# (memory efficient, does not copy strings)
	f.set_features([s])
	positions=DynamicIntArray()
	positions.append_element(0)
	positions.append_element(6)
	positions.append_element(16)
	positions.append_element(25)

	f.obtain_by_position_list(4,positions)
	#print(f.get_features())

	# now extract windows of size 8 from same positon list
	f.obtain_by_position_list(8,positions)
	#print(f.get_features())
	return f

if __name__=='__main__':
	print('Sliding Window')
	features_string_sliding_window_modular(*parameter_list[0])
