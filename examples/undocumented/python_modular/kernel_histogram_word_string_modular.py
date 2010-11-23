from tools.load import LoadMatrix
lm=LoadMatrix()

parameter_list=[[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),lm.load_labels('../data/label_train_dna.dat'),3,0,False],[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),lm.load_labels('../data/label_train_dna.dat'),3,0,False]]

def kernel_histogram_word_string_modular (fm_train_dna=lm.load_dna('../data/fm_train_dna.dat'),fm_test_dna=lm.load_dna('../data/fm_test_dna.dat'),label_train_dna=lm.load_labels('../data/label_train_dna.dat'),order=3,gap=0,reverse=False):
	print 'PluginEstimate w/ HistogramWord'
	from shogun.Features import StringCharFeatures, StringWordFeatures, DNA, Labels
	from shogun.Kernel import HistogramWordStringKernel
	from shogun.Classifier import PluginEstimate

	fm_train_dna=fm_train_dna
	fm_test_dna=fm_test_dna
	label_train_dna=label_train_dna
	order = order
	gap = gap
	reverse = reverse
	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	pie=PluginEstimate()
	labels=Labels(label_train_dna)
	pie.set_labels(labels)
	pie.set_features(feats_train)
	pie.train()

	kernel=HistogramWordStringKernel(feats_train, feats_train, pie)
	km_train=kernel.get_kernel_matrix()
	print km_train
	kernel.init(feats_train, feats_test)
	pie.set_features(feats_test)
	pie.classify().get_labels()
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')
	kernel_histogram_word_string_modular(*parameter_list[0])
