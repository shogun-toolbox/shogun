#!/usr/bin/env python
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format_modular (fname):
	import os
	from modshogun import SparseRealFeatures
	from modshogun import LibSVMFile

	f=SparseRealFeatures()
	lab=f.load_svmlight_file(LibSVMFile(fname))
	f.write_svmlight_file(LibSVMFile('testwrite.light', 'w'), lab)
#	os.unlink('testwrite.light')

if __name__=='__main__':
	print('Reading SVMLIGHT format')
	features_read_svmlight_format_modular(*parameter_list[0])
