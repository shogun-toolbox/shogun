#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2014 Soumyajit De
#

#!/usr/bin/env python

from tools.load import LoadMatrix

lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,2,0.75],[traindat,testdat,3,0.75]]

def kernel_ssk_string_modular (fm_train_dna=traindat, fm_test_dna=testdat, maxlen=1, decay=1):
	from modshogun import StringSubsequenceKernel
	from modshogun import StringCharFeatures, DNA

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_train_dna, DNA)

	kernel=StringSubsequenceKernel(feats_train, feats_train, maxlen, decay)

	km_train=kernel.get_kernel_matrix()
	# print(km_train)
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	# print(km_test)
	return km_train,km_test,kernel

if __name__=='__main__':
	print('StringSubsequenceKernel DNA')
	kernel_ssk_string_modular(*parameter_list[0])
	kernel_ssk_string_modular(*parameter_list[1])
