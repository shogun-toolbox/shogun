# Explicit examples on how to use distributions
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))


#
# distributions
#

# Histogram
print('Histogram')

#	sg('new_distribution', 'HISTOGRAM')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
dump <- sg('attach_preproc', 'TRAIN')

#	sg('train_distribution')
#	histo=sg('get_histogram')

#	num_examples=11
#	num_param=sg('get_histogram_num_model_parameters')
#	for i in xrange(num_examples):
#		for j in xrange(num_param):
#			sg('get_log_derivative %d %d' % (j, i))

#	sg('get_log_likelihood')
#	sg('get_log_likelihood_sample')
