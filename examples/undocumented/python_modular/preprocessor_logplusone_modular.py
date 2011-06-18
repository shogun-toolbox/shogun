from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat+10,testdat+10,1.4,10],[traindat+10,testdat+10,1.5,10]]

def preprocessor_logplusone_modular (fm_train_real=traindat,fm_test_real=testdat,width=1.4,size_cache=10):

	from shogun.Kernel import Chi2Kernel
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import LogPlusOne

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=LogPlusOne()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

if __name__=='__main__':
	print 'LogPlusOne'
	preprocessor_logplusone_modular(*parameter_list[0])
