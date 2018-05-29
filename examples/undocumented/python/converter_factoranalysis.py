#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data]]

def converter_factoranalysis(data_fname):
	try:
		import numpy
		from shogun import RealFeatures, FactorAnalysis, EuclideanDistance, CSVFile

		features = RealFeatures(CSVFile(data_fname))

		converter = FactorAnalysis()
		converter.set_target_dim(2)
		embedding = converter.transform(features)

		X = embedding.get_feature_matrix()
		covdet = numpy.linalg.det(numpy.dot(X,X.T))

		return covdet > 0
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('Factor Analysis')
	converter_factoranalysis(*parameter_list[0])
