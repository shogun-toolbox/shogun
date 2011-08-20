# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
# create string features with a single string
s='A'*10 + 'C'*10 + 'G'*10 + 'T'*10

parameter_list=[[s]]

def features_string_sliding_window_modular(strings)

# *** 	f=Modshogun::StringCharFeatures.new([strings])
	f=Modshogun::Modshogun.new
	f.set_features::StringCharFeatures.new([strings])

	# slide a window of length 5 over features
	# (memory efficient, does not copy strings)
	f.obtain_by_sliding_window(5,1)
	#	puts f.get_num_vectors()
	#	puts f.get_vector_length(0)
	#	puts f.get_vector_length(1)
	#	puts f.get_features()

	# slide a window of length 4 over features
	# (memory efficient, does not copy strings)
	f.obtain_by_sliding_window(4,1)
	#	puts f.get_num_vectors()
	#	puts f.get_vector_length(0)
	#	puts f.get_vector_length(1)
	#	puts f.get_features()

	# extract string-windows at position 0,6,16,25 of window size 4
	# (memory efficient, does not copy strings)
	f.set_features([s])
# *** 	positions=DynamicIntArray()
	positions=Modshogun::DynamicIntArray.new
	positions.set_features()
	positions.append_element(0)
	positions.append_element(6)
	positions.append_element(16)
	positions.append_element(25)

	f.obtain_by_position_list(4,positions)
	#	puts f.get_features()

	# now extract windows of size 8 from same positon list
	f.obtain_by_position_list(8,positions)
	#	puts f.get_features()
	return f


end
if __FILE__ == $0
	puts 'Sliding Window'
	features_string_sliding_window_modular(*parameter_list[0])

end
