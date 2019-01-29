#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 20], [data, 30]]

def converter_locallylinearembeeding (data, k):
	import shogun as sg

	features = sg.features(data)

	converter = sg.transformer('LocallyLinearEmbedding', k=k)

	converter.fit(features)
	features = converter.transform(features)

	return features


if __name__=='__main__':
	print('LocallyLinearEmbedding')
	converter_locallylinearembeeding(*parameter_list[0])

