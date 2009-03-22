def plugin_estimate_histogram ():
	print 'PluginEstimate w/ HistogramWord'
	from shogun.Features import StringCharFeatures, StringWordFeatures, DNA, Labels
	from shogun.Kernel import HistogramWordStringKernel
	from shogun.Classifier import PluginEstimate

	order=3
	gap=0
	reverse=False

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

	kernel.init(feats_train, feats_test)
	pie.set_features(feats_test)
	pie.classify().get_labels()
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')
	plugin_estimate_histogram()
