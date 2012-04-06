from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def converter_localtangentspacealignment_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Converter import LocalTangentSpaceAlignment
	
	features = RealFeatures(data)
		
	converter = LocalTangentSpaceAlignment()
	converter.set_target_dim(1)
	converter.set_k(k)
	converter.apply(features)

	return features


if __name__=='__main__':
	print('LocalTangentSpaceAlignment')
	converter_localtangentspacealignment_modular(*parameter_list[0])

