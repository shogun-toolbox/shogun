#!/usr/bin/env python
import numpy

matrix=numpy.array([[-1.0,0,1],[2,3,4],[5,6,7]])
bins=numpy.array([[0.0, 0.0, 0.0],[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0],[4.0,4.0,4.0]])


parameter_list = [(matrix,bins)]

def features_binned_dot_modular (matrix, bins):
	from modshogun import RealFeatures, BinnedDotFeatures
	rf=RealFeatures(matrix)

	#print(rf.get_feature_matrix())

	bf=BinnedDotFeatures(rf, bins)
	filled=bf.get_computed_dot_feature_matrix()

	bf.set_fill(False)
	unfilled=bf.get_computed_dot_feature_matrix()

	bf.set_norm_one(True)
	unfilled_normed=bf.get_computed_dot_feature_matrix()

	bf.set_fill(True)
	filled_normed=bf.get_computed_dot_feature_matrix()

	return bf,filled,unfilled,unfilled_normed,filled_normed

if __name__=='__main__':
    print('BinnedDotFeatures')
    features_binned_dot_modular(*parameter_list[0])
