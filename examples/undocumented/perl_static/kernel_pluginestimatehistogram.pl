from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
trainlabel=lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindna,testdna,trainlabel,10,3,0,'n'],
		[traindna,testdna,trainlabel,11,4,0,'n']]

def kernel_pluginestimatehistogram (fm_train_dna=traindna,fm_test_dna=testdna,
				 label_train_dna=trainlabel,size_cache=10,
				 order=3,gap=0,reverse='n',):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	pseudo_pos=1e-1
	pseudo_neg=1e-1
	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
	sg('set_labels', 'TRAIN', label_train_dna)
	sg('train_estimator')

	sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
	km=sg('get_kernel_matrix', 'TRAIN')

# not supported yet
#	lab=sg('plugin_estimate_classify')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('PluginEstimate w/ HistogramWord')
	kernel_pluginestimatehistogram(*parameter_list[0])
