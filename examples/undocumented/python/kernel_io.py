#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.9],[traindat,testdat,1.7]]

def kernel_io (train_fname=traindat,test_fname=testdat,width=1.9):
	from shogun import RealFeatures, GaussianKernel, CSVFile
	from tempfile import NamedTemporaryFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()
	tmp_train_csv = NamedTemporaryFile(suffix='train.csv')
	f=CSVFile(tmp_train_csv.name, "w")
	kernel.save(f)
	del f

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	tmp_test_csv = NamedTemporaryFile(suffix='test.csv')
	f=CSVFile(tmp_test_csv.name,"w")
	kernel.save(f)
	del f

	return km_train, km_test, kernel

if __name__=='__main__':
	print('Gaussian')
	kernel_io(*parameter_list[0])
