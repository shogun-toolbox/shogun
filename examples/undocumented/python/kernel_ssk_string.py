# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soumyajit De

#!/usr/bin/env python
import shogun as sg
from tools.load import LoadMatrix

lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,2,0.75],[traindat,testdat,3,0.75]]

def kernel_ssk_string (fm_train_dna=traindat, fm_test_dna=testdat, maxlen=1, decay=1):

	feats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_test=sg.create_string_features(fm_test_dna, sg.DNA)

	kernel=sg.create("SubsequenceStringKernel", maxlen=maxlen, decay=decay)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	# print(km_train)
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	# print(km_test)
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SubsequenceStringKernel DNA')
	kernel_ssk_string(*parameter_list[0])
	kernel_ssk_string(*parameter_list[1])
