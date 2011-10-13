from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def converter_multidimensionalscaling_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Converter import MultidimensionalScaling
	
	features = RealFeatures(data)
		
	converter = MultidimensionalScaling()
	converter.set_target_dim(1)
	converter.set_landmark(False)
	converter.apply(features)

	return features


if __name__=='__main__':
	print 'MultidimensionalScaling'
	converter_multidimensionalscaling_modular(*parameter_list[0])
