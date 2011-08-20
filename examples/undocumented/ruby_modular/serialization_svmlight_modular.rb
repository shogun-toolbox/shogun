# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
parameter_list=[[10, 1, 2.1, 2.0]]

def serialization_svmlight_modular(num, dist, width, C)

	import sys
	import types
	import random
	import bz2
	import cPickle as pickle
	import inspect


	def save(filename, myobj)
		"""
		save object to file using pickle
		
		@param filename: name of destination file
		@type filename: str
		@param myobj: object to save (has to be pickleable)
		@type myobj: obj
		"""

		try:
			f = bz2.BZ2File(filename, 'wb')
		except IOError, details:
			sys.stderr.write('File ' + filename + ' cannot be written\n')
			sys.stderr.write(details)
			return

		pickle.dump(myobj, f, protocol=2)
		f.close()



	def load(filename)
		"""
		Load from filename using pickle
		
		@param filename: name of file to load from
		@type filename: str
		"""
		
		try:
			f = bz2.BZ2File(filename, 'rb')
		except IOError, details:
			sys.stderr.write('File ' + filename + ' cannot be read\n')
			sys.stderr.write(details)
			return

		myobj = pickle.load(f)
		f.close()
		return myobj


	##################################################

	seed(17)
	traindata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
	testdata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

	trainlab=concatenate((-ones(num), ones(num)));
	testlab=concatenate((-ones(num), ones(num)));

# *** 	feats_train=RealFeatures(traindata_real);
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(traindata_real);
# *** 	feats_test=RealFeatures(testdata_real);
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(testdata_real);
# *** 	kernel=GaussianKernel(feats_train, feats_train, width);
	kernel=Modshogun::GaussianKernel.new
	kernel.set_features(feats_train, feats_train, width);
	#kernel.io.set_loglevel(MSG_DEBUG)

# *** 	labels=Labels(trainlab);
	labels=Modshogun::Labels.new
	labels.set_features(trainlab);

# *** 	svm=SVMLight(C, kernel, labels)
	svm=Modshogun::SVMLight.new
	svm.set_features(C, kernel, labels)
	svm.train()
	#svm.io.set_loglevel(MSG_DEBUG)

	##################################################

	#	puts "labels:"
	#	puts pickle.dumps(labels)
	#
	#	puts "features"
	#	puts pickle.dumps(feats_train)
	#
	#	puts "kernel"
	#	puts pickle.dumps(kernel)
	#
	#	puts "svm"
	#	puts pickle.dumps(svm)
	#
	#	puts "#################################"

	fn = "serialized_svm.bz2"
	#	puts "serializing SVM to file", fn

	save(fn, svm)

	#	puts "#################################"

	#	puts "unserializing SVM"
	svm2 = load(fn)


	#	puts "#################################"
	#	puts "comparing training"

	svm2.train()

	#	puts "objective before serialization:", svm.get_objective()
	#	puts "objective after serialization:", svm2.get_objective()
	return svm, svm.get_objective(), svm2, svm2.get_objective()


end
if __FILE__ == $0
	puts 'Serialization SVMLight'
	serialization_svmlight_modular(*parameter_list[0])

end
