#!/usr/bin/env python
parameter_list=[['../data/train_sparsereal.light']]

def features_read_svmlight_format (fname):
	from tempfile import NamedTemporaryFile
	from shogun import SparseRealFeatures
	from shogun import LibSVMFile

	f=SparseRealFeatures()
	lab=f.load_with_labels(LibSVMFile(fname))
	tmp_file = NamedTemporaryFile(suffix='svmlight')
	f.save_with_labels(LibSVMFile(tmp_file.name, 'w'), lab)

if __name__=='__main__':
	print('Reading SVMLIGHT format')
	features_read_svmlight_format(*parameter_list[0])
