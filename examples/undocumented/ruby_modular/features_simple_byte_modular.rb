# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
import numpy

# create dense matrix A
A=numpy.array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=numpy.uint8)

parameter_list=[[A]]

def features_simple_byte_modular(A)

	# create dense features a
	# ... of type Byte
# *** 	a=ByteFeatures(A)
	a=Modshogun::ByteFeatures.new
	a.set_features(A)

	#	puts some statistics about a
	#	puts a.get_num_vectors()
	#	puts a.get_num_features()

	# get first feature vector and set it
	#	puts a.get_feature_vector(0)
	a.set_feature_vector(numpy.array([1,4,0,0,0,9], dtype=numpy.uint8), 0)

	# get matrix
	a_out = a.get_feature_matrix()

	#	puts type(a_out), a_out.dtype
	#	puts a_out 
	assert(numpy.all(a_out==A))
	return a_out,a


end
if __FILE__ == $0
	puts 'ByteFeatures'
	features_simple_byte_modular(*parameter_list[0])

end
