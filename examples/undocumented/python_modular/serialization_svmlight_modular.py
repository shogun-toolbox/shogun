parameter_list=[[10, 1, 2.1, 2.0]]

def serialization_svmlight_modular(num, dist, width, C):
	from shogun.IO import MSG_DEBUG
	from shogun.Features import RealFeatures, Labels, DNA, Alphabet
	from shogun.Kernel import WeightedDegreeStringKernel, GaussianKernel
	from shogun.Classifier import SVMLight
	from numpy import concatenate, ones
	from numpy.random import randn, seed

	import sys
	import types
	import random
	import bz2
	try:
		import cPickle as pickle
	except ImportError:
		import pickle as pickle	
	import inspect


	def save(filename, myobj):
		"""
		save object to file using pickle
		
		@param filename: name of destination file
		@type filename: str
		@param myobj: object to save (has to be pickleable)
		@type myobj: obj
		"""

		try:
			f = bz2.BZ2File(filename, 'wb')
		except IOError as details:
			sys.stderr.write('File ' + filename + ' cannot be written\n')
			sys.stderr.write(details)
			return

		pickle.dump(myobj, f, protocol=2)
		f.close()



	def load(filename):
		"""
		Load from filename using pickle
		
		@param filename: name of file to load from
		@type filename: str
		"""
		
		try:
			f = bz2.BZ2File(filename, 'rb')
		except IOError as details:
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

	feats_train=RealFeatures(traindata_real);
	feats_test=RealFeatures(testdata_real);
	kernel=GaussianKernel(feats_train, feats_train, width);
	#kernel.io.set_loglevel(MSG_DEBUG)

	labels=Labels(trainlab);

	svm=SVMLight(C, kernel, labels)
	svm.train()
	#svm.io.set_loglevel(MSG_DEBUG)

	##################################################

	#print("labels:")
	#print(pickle.dumps(labels))
	#
	#print("features")
	#print(pickle.dumps(feats_train))
	#
	#print("kernel")
	#print(pickle.dumps(kernel))
	#
	#print("svm")
	#print(pickle.dumps(svm))
	#
	#print("#################################")

	fn = "serialized_svm.bz2"
	#print("serializing SVM to file", fn)

	save(fn, svm)

	#print("#################################")

	#print("unserializing SVM")
	svm2 = load(fn)


	#print("#################################")
	#print("comparing training")

	svm2.train()

	#print("objective before serialization:", svm.get_objective())
	#print("objective after serialization:", svm2.get_objective())
	return svm, svm.get_objective(), svm2, svm2.get_objective()

if __name__=='__main__':
	print('Serialization SVMLight')
	serialization_svmlight_modular(*parameter_list[0])
