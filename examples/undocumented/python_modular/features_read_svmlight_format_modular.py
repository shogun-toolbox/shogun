#!/usr/bin/env python
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format_modular (fname):
	import os
	from modshogun import SparseRealFeatures

	f=SparseRealFeatures()
	lab=f.load_svmlight_file(fname)
	f.write_svmlight_file('testwrite.light', lab)
	os.unlink('testwrite.light')

if __name__=='__main__':
	print('Reading SVMLIGHT format')
	features_read_svmlight_format_modular(*parameter_list[0])
