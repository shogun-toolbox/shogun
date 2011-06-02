parameter_list=[[5,1,10, 2.0, 10], [10,0.3,2, 1.0, 0.1]]

def check_status(status):
	# silent...
	assert(status)
	#if  status:
	#	print "OK reading/writing .h5\n"
	#else:
	#	print "ERROR reading/writing .h5\n"

def serialization_complex_example(num=5, dist=1, dim=10, C=2.0, width=10):
	import os
	from numpy import concatenate, zeros, ones
	from numpy.random import randn, seed
	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import GMNPSVM
	from shogun.Kernel import GaussianKernel
	from shogun.Library import SerializableHdf5File,SerializableAsciiFile, \
			SerializableJsonFile,SerializableXmlFile,MSG_DEBUG
	from shogun.Preprocessor import NormOne, LogPlusOne

	seed(17)

	data=concatenate((randn(dim, num), randn(dim, num) + dist,
					  randn(dim, num) + 2*dist,
					  randn(dim, num) + 3*dist), axis=1)
	lab=concatenate((zeros(num), ones(num), 2*ones(num), 3*ones(num)))

	feats=RealFeatures(data)
	#feats.io.set_loglevel(MSG_DEBUG)
	kernel=GaussianKernel(feats, feats, width)

	labels=Labels(lab)

	svm = GMNPSVM(C, kernel, labels)

	feats.add_preproc(NormOne())
	feats.add_preproc(LogPlusOne())
	feats.set_preprocessed(1)
	svm.train(feats)

	#svm.print_serializable()

	fstream = SerializableHdf5File("blaah.h5", "w")
	status = svm.save_serializable(fstream)
	check_status(status)

	fstream = SerializableAsciiFile("blaah.asc", "w")
	status = svm.save_serializable(fstream)
	check_status(status)

	fstream = SerializableJsonFile("blaah.json", "w")
	status = svm.save_serializable(fstream)
	check_status(status)

	fstream = SerializableXmlFile("blaah.xml", "w")
	status = svm.save_serializable(fstream)
	check_status(status)


	fstream = SerializableHdf5File("blaah.h5", "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status)
	new_svm.train()

	fstream = SerializableAsciiFile("blaah.asc", "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status)
	new_svm.train()

	fstream = SerializableJsonFile("blaah.json", "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status)
	new_svm.train()

	fstream = SerializableXmlFile("blaah.xml", "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status)
	new_svm.train()

	os.unlink("blaah.h5")
	os.unlink("blaah.asc")
	os.unlink("blaah.json")
	os.unlink("blaah.xml")
	return svm,new_svm


if __name__=='__main__':
	print 'Serialization SVMLight'
	serialization_complex_example(*parameter_list[0])
