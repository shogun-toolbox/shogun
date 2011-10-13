from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_isomap_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Converter import Isomap
	
	features = RealFeatures(data)
		
	converter = Isomap()
	converter.set_landmark(True)
	converter.set_landmark_number(5)
	converter.set_k(6)
	converter.set_target_dim(1)
	converter.apply(features)

	return features


if __name__=='__main__':
	print 'Isomap'
	converter_isomap_modular(*parameter_list[0])

