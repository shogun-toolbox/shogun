# Explicit examples on how to use distributions
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30
size_cache <- 10
order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true

getDNA <- function(len, num) {
	acgt <- c('A', 'C', 'G', 'T')
	data <- c()
	for (j in 1:num) {
		str <- '';
		for (i in 1:len) {
			str <- paste(str, sample(acgt, 1), sep='')
		}
		data <- append(data, str)
	}
	data
}

traindat_dna <- getDNA(len, num)
testdat_dna <- getDNA(len, num+7)

#
# distributions
#

# Histogram
print('Histogram')

#	sg('new_distribution', 'HISTOGRAM')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
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


# Linear HMM
print('LinearHMM')

#	sg('new_distribution', 'LinearHMM')
dump <- sg('add_preproc', 'SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
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


# HMM
print('HMM')

N <- 3
M <- 6
order <- 1
hmms <- c()
liks <- c()

cube <- list(NULL, NULL, NULL)
num <- vector(mode='numeric',length=18)+100
num[1] <- 0;
num[2] <- 0;
num[3] <- 0;
num[10] <- 0;
num[11] <- 0;
num[12] <- 0;

for (c in 1:3)
{
	for (i in 1:6)
	{
		cube[[c]] <- c(cube[[c]], vector(mode='numeric',length=num[(c-1)*6+i])+i)
	}
	cube[[c]] <- sample(cube[[c]],300,replace=TRUE);
}

cube <- c(cube[[1]], cube[[2]], cube[[3]])
cube <- paste(cube, sep="", collapse="")

dump <- sg('set_features', 'TRAIN', cube, 'CUBE')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order)

dump <- sg('new_hmm', N, M)
dump <- sg('bw')
hmm <- sg('get_hmm')

dump <- sg('new_hmm', N, M)
dump <- sg('set_hmm', hmm[[1]], hmm[[2]], hmm[[3]], hmm[[4]])
likelihood <- sg('hmm_likelihood')

