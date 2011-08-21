# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3,0,False],[traindna,4,0,False]]

# *** def distribution_histogram_modular(fm_dna=traindna,order=3,gap=0,reverse=False)
def distribution_histogram_modular(fm_dna=traindna,order=3,gap=0,reverse=Modshogun::False.new
def distribution_histogram_modular(fm_dna=traindna,order=3,gap=0,reverse.set_features)

# *** 	charfeat=StringCharFeatures(DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(DNA)
	charfeat.set_features(fm_dna)
# *** 	feats=StringWordFeatures(charfeat.get_alphabet())
	feats=Modshogun::StringWordFeatures.new
	feats.set_features(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

# *** 	histo=Histogram(feats)
	histo=Modshogun::Histogram.new
	histo.set_features(feats)
	histo.train()

	histo.get_histogram()

	num_examples=feats.get_num_vectors()
	num_param=histo.get_num_model_parameters()
	#for i in xrange(num_examples):
	#	for j in xrange(num_param):
	#		histo.get_log_derivative(j, i)

	out_likelihood = histo.get_log_likelihood()
	out_sample = histo.get_log_likelihood_sample()
	return histo,out_sample,out_likelihood

end
###########################################################################
# call functions
###########################################################################

if __FILE__ == $0
	puts 'Histogram'
	distribution_histogram_modular(*parameter_list[0])


end
