#!/usr/bin/env python
data = '../data/fm_train_real.dat'
parameter_list = [[data]]

def converter_multidimensionalscaling_modular (data_fname):
	try:
		import numpy
		from modshogun import RealFeatures, MultidimensionalScaling, EuclideanDistance, CSVFile
		
		features = RealFeatures(CSVFile(data_fname))
			
		distance_before = EuclideanDistance()
		distance_before.init(features,features)

		converter = MultidimensionalScaling()
		converter.set_target_dim(2)
		converter.set_landmark(False)
		embedding = converter.apply(features)

		distance_after = EuclideanDistance()
		distance_after.init(embedding,embedding)

		distance_matrix_after = distance_after.get_distance_matrix()
		distance_matrix_before = distance_before.get_distance_matrix()

		return numpy.linalg.norm(distance_matrix_after-distance_matrix_before)/numpy.linalg.norm(distance_matrix_before) < 1e-6
	except ImportError:
		print('No Eigen3 available')

if __name__=='__main__':
	print('MultidimensionalScaling')
	converter_multidimensionalscaling_modular(*parameter_list[0])
