require 'modshogun'
require 'pp'
require 'load'

traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna, 3, 0, false], [traindna, 4, 0, false]]

def distribution_histogram_modular(fm_dna=traindna, order=3, gap=0, reverse=false)

	charfeat=Modshogun::StringCharFeatures.new(Modshogun::DNA)
	charfeat.set_features(fm_dna)
	feats=Modshogun::StringWordFeatures.new(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)

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
	pp distribution_histogram_modular(*parameter_list[0])
end
