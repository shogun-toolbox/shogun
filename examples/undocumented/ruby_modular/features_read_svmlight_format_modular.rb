# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format_modular(fname)
	import os

# *** 	f=SparseRealFeatures()
	f=Modshogun::SparseRealFeatures.new
	f.set_features()
	lab=f.load_svmlight_file(fname)
	f.write_svmlight_file('testwrite.light', lab)
	os.unlink('testwrite.light')


end
if __FILE__ == $0
	puts 'Reading SVMLIGHT format'
	features_read_svmlight_format_modular(*parameter_list[0])

end
