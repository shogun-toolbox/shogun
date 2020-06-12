#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.9],[traindat,testdat,1.7]]

def kernel_io (train_fname=traindat,test_fname=testdat,width=1.9):
	from tempfile import NamedTemporaryFile
	import shogun as sg

	feats_train=sg.create_features(sg.read_csv(train_fname))
	feats_test=sg.create_features(sg.read_csv(test_fname))

	kernel=sg.create_kernel("GaussianKernel", log_width=width)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	tmp_train_csv = NamedTemporaryFile(suffix='train.csv')
	f=sg.read_csv(tmp_train_csv.name, "w")
	kernel.save(f)
	del f

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	tmp_test_csv = NamedTemporaryFile(suffix='test.csv')
	f=sg.read_csv(tmp_test_csv.name,"w")
	kernel.save(f)
	del f

	return km_train, km_test, kernel

if __name__=='__main__':
	print('Gaussian')
	kernel_io(*parameter_list[0])
