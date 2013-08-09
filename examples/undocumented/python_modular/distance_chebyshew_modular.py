#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat],[traindat,testdat]]

def distance_chebyshew_modular (train_fname=traindat,test_fname=testdat):
	from modshogun import RealFeatures, ChebyshewMetric, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance=ChebyshewMetric(feats_train, feats_train)
	dm_train=distance.get_distance_matrix()

	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

if __name__=='__main__':
	print('ChebyshewMetric')
	distance_chebyshew_modular(*parameter_list[0])
