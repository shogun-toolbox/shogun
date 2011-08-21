# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list=[['../data/snps.dat']]

def features_snp_modular(fname)

# *** 	sf=StringByteFeatures(SNP)
	sf=Modshogun::StringByteFeatures.new
	sf.set_features(SNP)
	sf.load_ascii_file(fname, False, SNP, SNP)
	#	puts sf.get_features()
# *** 	snps=SNPFeatures(sf)
	snps=Modshogun::SNPFeatures.new
	snps.set_features(sf)
	#	puts snps.get_feature_matrix()
	#	puts snps.get_minor_base_string()
	#	puts snps.get_major_base_string()


end
if __FILE__ == $0
	puts 'SNP Features'
	features_snp_modular(*parameter_list[0])

end
