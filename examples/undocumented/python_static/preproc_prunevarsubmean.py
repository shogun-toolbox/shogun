from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.4,10,True],[traindat,testdat,1.5,11,True]]

def preproc_prunevarsubmean (fm_train_real=traindat,fm_test_real=testdat,
		 width=1.4,size_cache=10,divide_by_std=True):

	sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std)
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	km=sg('get_kernel_matrix', 'TRAIN')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('PruneVarSubMean')
	preproc_prunevarsubmean(*parameter_list[0])
