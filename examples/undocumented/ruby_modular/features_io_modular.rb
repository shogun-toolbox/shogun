# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
data=LoadMatrix.load_numbers('../data/fm_train_real.dat')
label=LoadMatrix.load_numbers('../data/label_train_twoclass.dat')

parameter_list=[[data,label]]

def features_io_modular(fm_train_real, label_train_twoclass)

# *** 	feats=SparseRealFeatures(fm_train_real)
	feats=Modshogun::SparseRealFeatures.new
	feats.set_full_feature_matrix(fm_train_real)
# *** 	feats2=SparseRealFeatures()
	feats2=Modshogun::SparseRealFeatures.new

# *** 	f=BinaryFile("fm_train_sparsereal.bin","w")
	f=Modshogun::BinaryFile.new("fm_train_sparsereal.bin","w")
	#f.set_features("fm_train_sparsereal.bin","w")
	feats.save(f)

# *** 	f=AsciiFile("fm_train_sparsereal.ascii","w")
	f=Modshogun::AsciiFile.new("fm_train_sparsereal.ascii","w")
	#f.set_features("fm_train_sparsereal.ascii","w")
	feats.save(f)

# *** 	f=BinaryFile("fm_train_sparsereal.bin")
	f=Modshogun::BinaryFile.new("fm_train_sparsereal.bin")
	#f.set_features("fm_train_sparsereal.bin")
	feats2.load(f)

# *** 	f=AsciiFile("fm_train_sparsereal.ascii")
	f=Modshogun::AsciiFile.new("fm_train_sparsereal.ascii")
	#f.set_features("fm_train_sparsereal.ascii")
	feats2.load(f)

# *** 	feats=RealFeatures(fm_train_real)
	feats=Modshogun::RealFeatures.new
	feats.set_feature_matrix(fm_train_real)
# *** 	feats2=RealFeatures()
	feats2=Modshogun::RealFeatures.new
	#feats2.set_features()

# *** 	f=BinaryFile("fm_train_real.bin","w")
	f=Modshogun::BinaryFile.new("fm_train_real.bin","w")
	#f.set_features("fm_train_real.bin","w")
	feats.save(f)

# *** 	f=HDF5File("fm_train_real.h5","w", "/data/doubles")
	f=Modshogun::HDF5File.new("fm_train_real.h5","w", "/data/doubles")
	#f.set_features("fm_train_real.h5","w", "/data/doubles")
	feats.save(f)

# *** 	f=AsciiFile("fm_train_real.ascii","w")
	f=Modshogun::AsciiFile.new("fm_train_real.ascii","w")
	#f.set_features("fm_train_real.ascii","w")
	feats.save(f)

# *** 	f=BinaryFile("fm_train_real.bin")
	f=Modshogun::BinaryFile.new("fm_train_real.bin")
	#f.set_features("fm_train_real.bin")
	feats2.load(f)
	#	puts "diff binary", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten()))

# *** 	f=AsciiFile("fm_train_real.ascii")
	f=Modshogun::AsciiFile.new
	#f.set_features("fm_train_real.ascii")
	feats2.load(f)
	#	puts "diff ascii", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten()))

# *** 	lab=Labels(numpy.array([1.0,2.0,3.0]))
	lab=Modshogun::Labels.new
	lab.set_features(numpy.array([1.0,2.0,3.0]))
# *** 	lab2=Labels()
	lab2=Modshogun::Labels.new
	lab2.set_features()
# *** 	f=AsciiFile("label_train_twoclass.ascii","w")
	f=Modshogun::AsciiFile.new
	f.set_features("label_train_twoclass.ascii","w")
	lab.save(f)

# *** 	f=BinaryFile("label_train_twoclass.bin","w")
	f=Modshogun::BinaryFile.new
	f.set_features("label_train_twoclass.bin","w")
	lab.save(f)

# *** 	f=HDF5File("label_train_real.h5","w", "/data/labels")
	f=Modshogun::HDF5File.new
	f.set_features("label_train_real.h5","w", "/data/labels")
	lab.save(f)

# *** 	f=AsciiFile("label_train_twoclass.ascii")
	f=Modshogun::AsciiFile.new
	f.set_features("label_train_twoclass.ascii")
	lab2.load(f)

# *** 	f=BinaryFile("label_train_twoclass.bin")
	f=Modshogun::BinaryFile.new
	f.set_features("label_train_twoclass.bin")
	lab2.load(f)

# *** 	f=HDF5File("fm_train_real.h5","r", "/data/doubles")
	f=Modshogun::HDF5File.new
	f.set_features("fm_train_real.h5","r", "/data/doubles")
	feats2.load(f)
	#	puts feats2.get_feature_matrix()
# *** 	f=HDF5File("label_train_real.h5","r", "/data/labels")
	f=Modshogun::HDF5File.new
	f.set_features("label_train_real.h5","r", "/data/labels")
	lab2.load(f)
	#	puts lab2.get_labels()

	#clean up
	import os
	for f in ['fm_train_sparsereal.bin','fm_train_sparsereal.ascii',
			'fm_train_real.bin','fm_train_real.h5','fm_train_real.ascii',
			'label_train_real.h5', 'label_train_twoclass.ascii','label_train_twoclass.bin']:
		os.unlink(f)
	end
	return feats, feats2, lab, lab2
end

if __FILE__ == $0
	puts 'Features IO'
	pp features_io_modular(*parameter_list[0])
end
