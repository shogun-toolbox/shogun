from tools.load import LoadMatrix
import numpy

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_multidimensionalscaling_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Converter import MultidimensionalScaling
	from shogun.Distance import EuclidianDistance
	
	features = RealFeatures(data)
		
	distance_before = EuclidianDistance()
	distance_before.init(features,features)

	converter = MultidimensionalScaling()
	converter.set_target_dim(2)
	converter.set_landmark(False)
	embedding =converter.apply(features)

	distance_after = EuclidianDistance()
	distance_after.init(embedding,embedding)

	distance_matrix_after = distance_after.get_distance_matrix()
	distance_matrix_before = distance_before.get_distance_matrix()

	return numpy.linalg.norm(distance_matrix_after-distance_matrix_before)/numpy.linalg.norm(distance_matrix_before)

if __name__=='__main__':
	print('MultidimensionalScaling')
	converter_multidimensionalscaling_modular(*parameter_list[0])
