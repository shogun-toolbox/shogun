# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

parameter_list=[[[[1.0,2,3],[4,5,6]]]]

def serialization_matrix_modular(m)
# *** 	feats=RealFeatures(array(m))
	feats=Modshogun::RealFeatures.new
	feats.set_features(array(m))
	#feats.io.set_loglevel(0)
	fstream = SerializableAsciiFile("foo.asc", "w")
	feats.save_serializable(fstream)

# *** 	l=Labels(array([1.0,2,3]))
	l=Modshogun::Labels.new
	l.set_features(array([1.0,2,3]))
	fstream = SerializableAsciiFile("foo2.asc", "w")
	l.save_serializable(fstream)

	os.unlink("foo.asc")
	os.unlink("foo2.asc")


end
if __FILE__ == $0
	puts 'Serialization Matrix Modular'
	serialization_matrix_modular(*parameter_list[0])

end
