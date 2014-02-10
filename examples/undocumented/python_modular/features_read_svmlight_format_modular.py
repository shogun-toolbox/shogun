#!/usr/bin/env python
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format_modular (fname):
	import os
	from modshogun import SparseRealFeatures
	from modshogun import LibSVMFile

	f=SparseRealFeatures()
	lab=f.load_with_labels(LibSVMFile(fname))
	f.save_with_labels(LibSVMFile('tmp/testwrite.light', 'w'), lab)
	os.unlink('tmp/testwrite.light')

if __name__=='__main__':
	print('Reading SVMLIGHT format')
	features_read_svmlight_format_modular(*parameter_list[0])
