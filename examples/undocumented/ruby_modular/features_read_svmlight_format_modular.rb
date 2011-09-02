require 'modshogun'
require 'pp'
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format_modular(fname)

	f=Modshogun::SparseRealFeatures.new
	lab=f.load_svmlight_file(fname)
	f.write_svmlight_file('testwrite.light', lab)

end

if __FILE__ == $0
	puts 'Reading SVMLIGHT format'
	pp features_read_svmlight_format_modular(*parameter_list[0])
end
