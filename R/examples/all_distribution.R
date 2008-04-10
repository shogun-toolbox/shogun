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

#	sg('send_command', 'new_distribution HISTOGRAM')
dump <- sg('send_command', 'add_preproc SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('send_command', paste('convert TRAIN STRING CHAR STRING WORD', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TRAIN')

#	sg('send_command', 'train_distribution')
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

#	sg('send_command', 'new_distribution LinearHMM')
dump <- sg('send_command', 'add_preproc SORTWORDSTRING')

dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('send_command', paste('convert TRAIN STRING CHAR STRING WORD', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TRAIN')

#	sg('send_command', 'train_distribution')
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
dump <- sg('send_command', 'convert TRAIN STRING CHAR STRING WORD 1')

dump <- sg('send_command', paste('new_hmm', N, M))
dump <- sg('send_command', 'bw')
hmm <- sg('get_hmm')

dump <- sg('send_command', paste('new_hmm', N, M))
dump <- sg('set_hmm', hmm[[1]], hmm[[2]], hmm[[3]], hmm[[4]])
likelihood <- sg('hmm_likelihood')

