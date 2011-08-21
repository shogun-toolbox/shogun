# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list = [['features_string_char_compressed_modular.py']]

def features_string_char_compressed_modular(fname)

# *** 	f=StringFileCharFeatures(fname, RAWBYTE)
	f=Modshogun::StringFileCharFeatures.new
	f.set_features(fname, RAWBYTE)

	#	puts "original strings", f.get_features()

	#uncompressed
	f.save_compressed("foo_uncompressed.str", UNCOMPRESSED, 1)
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_uncompressed.str", True)
	#	puts "uncompressed strings", f2.get_features()
	#	puts 
	# load compressed data and uncompress on load

	#snappy - not stable yet?!
	#f.save_compressed("foo_snappy.str", SNAPPY, 9)
# *** 	#f2=StringCharFeatures(RAWBYTE);
	#f2=Modshogun::StringCharFeatures.new
	#f2.set_features(RAWBYTE);
	#f2.load_compressed("foo_snappy.str", True)
	#	puts "snappy strings", f2.get_features()
	#	puts 
	#lzo
	f.save_compressed("foo_lzo.str", LZO, 9)
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_lzo.str", True)
	#	puts "lzo strings", f2.get_features()
	#	puts 
	##gzip
	f.save_compressed("foo_gzip.str", GZIP, 9)
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_gzip.str", True)
	#	puts "gzip strings", f2.get_features()
	#	puts 
	#bzip2
	f.save_compressed("foo_bzip2.str", BZIP2, 9)
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_bzip2.str", True)
	#	puts "bzip2 strings", f2.get_features()
	#	puts 
	#lzma
	f.save_compressed("foo_lzma.str", LZMA, 9)
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_lzma.str", True)
	#	puts "lzma strings", f2.get_features()
	#	puts 
	# load compressed data and uncompress via preprocessor
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_lzo.str", False)
	f2.add_preprocessor(DecompressCharString(LZO))
	f2.apply_preprocessor()
	#	puts "lzo strings", f2.get_features()
	#	puts 
	# load compressed data and uncompress on-the-fly via preprocessor
# *** 	f2=StringCharFeatures(RAWBYTE);
	f2=Modshogun::StringCharFeatures.new
	f2.set_features(RAWBYTE);
	f2.load_compressed("foo_lzo.str", False)
	#f2.io.set_loglevel(MSG_DEBUG)
	f2.add_preprocessor(DecompressCharString(LZO))
	f2.enable_on_the_fly_preprocessing()
	#	puts "lzo strings", f2.get_features()
	#	puts 
	#clean up
	import os
	for f in ['foo_uncompressed.str', 'foo_snappy.str', 'foo_lzo.str', 'foo_gzip.str',
	'foo_bzip2.str', 'foo_lzma.str', 'foo_lzo.str', 'foo_lzo.str']:
		if os.path.exists(f):
			os.unlink(f)

	##########################################################################################
	# some perfectly compressible stuff follows
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################
	##########################################################################################


end
if __FILE__ == $0
	puts 'Compressing StringCharFileFeatures'
    features_string_char_compressed_modular(*parameter_list[0])

end
