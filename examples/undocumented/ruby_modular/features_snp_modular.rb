# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
parameter_list=[['../data/snps.dat']]

def features_snp_modular(fname)

	sf=StringByteFeatures(SNP)
	sf.load_ascii_file(fname, False, SNP, SNP)
	#print sf.get_features()
	snps=SNPFeatures(sf)
	#print snps.get_feature_matrix()
	#print snps.get_minor_base_string()
	#print snps.get_major_base_string()


end
if __FILE__ == $0
	print 'SNP Features'
	features_snp_modular(*parameter_list[0])

end
