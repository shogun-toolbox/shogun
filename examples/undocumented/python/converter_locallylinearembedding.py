#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 20], [data, 30]]

def converter_locallylinearembeeding (data, k):
	from shogun import RealFeatures
	from shogun import LocallyLinearEmbedding

	features = RealFeatures(data)

	converter = LocallyLinearEmbedding()
	converter.set_k(k)

	converter.fit(features)
	features = converter.apply(features)

	return features


if __name__=='__main__':
	print('LocallyLinearEmbedding')
	converter_locallylinearembeeding(*parameter_list[0])

