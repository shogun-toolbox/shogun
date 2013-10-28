#!/usr/bin/env python
parameter_list=[['../data/snps.dat']]

def features_snp_modular (fname):
	from modshogun import StringByteFeatures, SNPFeatures, SNP

	sf=StringByteFeatures(SNP)
	sf.load_ascii_file(fname, False, SNP, SNP)
	#print(sf.get_features())
	snps=SNPFeatures(sf)
	#print(snps.get_feature_matrix())
	#print(snps.get_minor_base_string())
	#print(snps.get_major_base_string())

if __name__=='__main__':
	print('SNP Features')
	features_snp_modular(*parameter_list[0])
