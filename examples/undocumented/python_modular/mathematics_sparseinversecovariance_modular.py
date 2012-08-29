#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,0.0],[data,1.0]]

def mathematics_sparseinversecovariance_modular (data,lc):
	from modshogun import SparseInverseCovariance
	from numpy import dot

	sic = SparseInverseCovariance()
	S = dot(data,data.T)
	Si = sic.estimate(S,lc)

	return Si


if __name__=='__main__':
	print('SparseInverseCovariance')
	mathematics_sparseinversecovariance_modular(*parameter_list[0])


