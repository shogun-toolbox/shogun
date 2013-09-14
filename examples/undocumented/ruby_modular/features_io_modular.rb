require 'load'
require 'modshogun'
require 'pp'

data=LoadMatrix.load_numbers('../data/fm_train_real.dat')
label=LoadMatrix.load_numbers('../data/label_train_twoclass.dat')

parameter_list=[[data,label]]

def features_io_modular(fm_train_real, label_train_twoclass)

	feats=Modshogun::SparseRealFeatures.new
	feats.set_full_feature_matrix(fm_train_real)
	feats2=Modshogun::SparseRealFeatures.new

	f=Modshogun::BinaryFile.new("fm_train_sparsereal.bin","w")
	feats.save(f)
	f.close()

	f=Modshogun::LibSVMFile.new("fm_train_sparsereal.ascii","w")
	feats.save(f)
	f.close()

	f=Modshogun::BinaryFile.new("fm_train_sparsereal.bin", "r")
	feats2.load(f)
	f.close()

	f=Modshogun::LibSVMFile.new("fm_train_sparsereal.ascii")
	feats2.load(f)
	f.close()

	feats=Modshogun::RealFeatures.new
	feats.set_feature_matrix(fm_train_real)
	feats2=Modshogun::RealFeatures.new

	f=Modshogun::BinaryFile.new("fm_train_real.bin","w")
	feats.save(f)
	f.close()

	f=Modshogun::HDF5File.new("fm_train_real.h5","w", "/data/doubles")
	feats.save(f)
	f.close()

	f=Modshogun::CSVFile.new("fm_train_real.ascii","w")
	feats.save(f)
	f.close()

	f=Modshogun::BinaryFile.new("fm_train_real.bin")
	feats2.load(f)
	f.close()

	f=Modshogun::CSVFile.new("fm_train_real.ascii")
	feats2.load(f)
	f.close()

	lab=Modshogun::MulticlassLabels.new([0.0,1.0,2.0,3.0])
	lab2=Modshogun::MulticlassLabels.new

	f=Modshogun::CSVFile.new("label_train_twoclass.ascii","w")
	lab.save(f)
	f.close()

	f=Modshogun::BinaryFile.new("label_train_twoclass.bin","w")
	lab.save(f)
	f.close()

	f=Modshogun::HDF5File.new("label_train_real.h5","w", "/data/labels")
	lab.save(f)
	f.close()

	f=Modshogun::CSVFile.new("label_train_twoclass.ascii")
	lab2.load(f)
	f.close()

	f=Modshogun::BinaryFile.new("label_train_twoclass.bin")
	lab2.load(f)
	f.close()

	f=Modshogun::HDF5File.new("fm_train_real.h5","r", "/data/doubles")
	feats2.load(f)
	f.close()

	f=Modshogun::HDF5File.new("label_train_real.h5","r", "/data/labels")
	lab2.load(f)
	f.close()
	#pp lab.get_labels()

	##clean up
	#import os
	#for f in ['fm_train_sparsereal.bin','fm_train_sparsereal.ascii',
	#		'fm_train_real.bin','fm_train_real.h5','fm_train_real.ascii',
	#		'label_train_real.h5', 'label_train_twoclass.ascii','label_train_twoclass.bin']:
	#	os.unlink(f)
	#end
	
	return feats, feats2, lab, lab2
end

if __FILE__ == $0
	puts 'Features IO'
	pp features_io_modular(*parameter_list[0])
end
