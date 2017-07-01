#!/usr/bin/env python
import numpy
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-3],[traindat,testdat,label_traindat,0.8,1e-2]]

def features_director_dot_modular (fm_train_real, fm_test_real,
		label_train_twoclass, C, epsilon):
	try:
		from modshogun import DirectorDotFeatures
		from modshogun import RealVector
	except ImportError:
		print("recompile shogun with --enable-swig-directors")
		return

	class NumpyFeatures(DirectorDotFeatures):

		# variables
		data=numpy.empty((1,1))

		# constructor
		def __init__(self, d):
			DirectorDotFeatures.__init__(self)
			self.data = d

		# overloaded methods
		def add_to_dense_sgvec(self, alpha, vec_idx1, vec2, abs):
			if abs:
				vec2+=alpha*numpy.abs(self.data[:,vec_idx1])
			else:
				vec2+=alpha*self.data[:,vec_idx1]

		def dot(self, vec_idx1, df, vec_idx2):
			return numpy.dot(self.data[:,vec_idx1], df.get_computed_dot_feature_vector(vec_idx2))

		def dense_dot_sgvec(self, vec_idx1, vec2):
			return numpy.dot(self.data[:,vec_idx1], vec2[0:vec2.vlen])

		def get_num_vectors(self):
			return self.data.shape[1]

		def get_dim_feature_space(self):
			return self.data.shape[0]

		# operators
	#	def __add__(self, other):
	#		return NumpyFeatures(self.data+other.data)

	#	def __sub__(self, other):
	#		return NumpyFeatures(self.data-other.data)

	#	def __iadd__(self, other):
	#		return NumpyFeatures(self.data+other.data)

	#	def __isub__(self, other):
	#		return NumpyFeatures(self.data-other.data)


	#from modshogun import RealFeatures, SparseRealFeatures, BinaryLabels
	#from modshogun import LibLinear, L2R_L2LOSS_SVC_DUAL
	#from modshogun import Math_init_random
	#Math_init_random(17)

	#feats_train=RealFeatures(fm_train_real)
	#feats_test=RealFeatures(fm_test_real)
	#labels=BinaryLabels(label_train_twoclass)

	#dfeats_train=NumpyFeatures(fm_train_real)
	#dfeats_test=NumpyFeatures(fm_test_real)
	#dlabels=BinaryLabels(label_train_twoclass)

	#print feats_train.get_computed_dot_feature_matrix()
	#print dfeats_train.get_computed_dot_feature_matrix()

	#svm=LibLinear(C, feats_train, labels)
	#svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)
	#svm.set_epsilon(epsilon)
	#svm.set_bias_enabled(True)
	#svm.train()

	#svm.set_features(feats_test)
	#svm.apply().get_labels()
	#predictions = svm.apply()

	#dfeats_train.__disown__()
	#dfeats_train.parallel.set_num_threads(1)
	#dsvm=LibLinear(C, dfeats_train, dlabels)
	#dsvm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)
	#dsvm.set_epsilon(epsilon)
	#dsvm.set_bias_enabled(True)
	#dsvm.train()

	#dfeats_test.__disown__()
	#dfeats_test.parallel.set_num_threads(1)
	#dsvm.set_features(dfeats_test)
	#dsvm.apply().get_labels()
	#dpredictions = dsvm.apply()

	#return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('DirectorLinear')
	features_director_dot_modular(*parameter_list[0])
