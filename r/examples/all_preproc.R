# Explicit examples on how to use the different preprocs
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30
size_cache <- 10

#
# real features
#

traindat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
testdat_real <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
width <- 1.4


# LogPlusOne
print('LogPlusOne')

dump <- sg('send_command', 'add_preproc LOGPLUSONE')
dump <- sg('send_command', paste('set_kernel CHI2 REAL', size_cache, width))

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', 'attach_preproc TRAIN')
dump <- sg('send_command', 'init_kernel TRAIN')
km <- sg('get_kernel_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'attach_preproc TEST')
dump <- sg('send_command', 'init_kernel TEST')
km <- sg('get_kernel_matrix')


# NormOne
print('NormOne')

dump <- sg('send_command', 'add_preproc NORMONE')
dump <- sg('send_command', paste('set_kernel CHI2 REAL', size_cache, width))

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', 'attach_preproc TRAIN')
dump <- sg('send_command', 'init_kernel TRAIN')
km <- sg('get_kernel_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'attach_preproc TEST')
dump <- sg('send_command', 'init_kernel TEST')
km <- sg('get_kernel_matrix')


# PruneVarSubMean
print('PruneVarSubMean')

divide_by_std <- 1
dump <- sg('send_command', paste('add_preproc PRUNEVARSUBMEAN', divide_by_std))
dump <- sg('send_command', paste('set_kernel CHI2 REAL', size_cache, width))

dump <- sg('set_features', 'TRAIN', traindat_real)
dump <- sg('send_command', 'attach_preproc TRAIN')
dump <- sg('send_command', 'init_kernel TRAIN')
km <- sg('get_kernel_matrix')

dump <- sg('set_features', 'TEST', testdat_real)
dump <- sg('send_command', 'attach_preproc TEST')
dump <- sg('send_command', 'init_kernel TEST')
km <- sg('get_kernel_matrix')


#
# complex string features
#

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

order <- 3
gap <- 0
reverse <- 'n' # bit silly to not use boolean, set 'r' to yield true
use_sign <- 0
normalization <- 'FULL'


# Comm Word String
print('CommWordString')

dump <- sg('send_command', 'add_preproc SORTWORDSTRING')
dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('send_command', paste('convert TRAIN STRING CHAR STRING WORD', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TRAIN')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('send_command', paste('convert TEST STRING CHAR STRING WORD', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TEST')

dump <- sg('send_command', paste('set_kernel COMMSTRING WORD', size_cache, use_sign, normalization))
dump <- sg('send_command', 'init_kernel TRAIN')
km <- sg('get_kernel_matrix')

dump <- sg('send_command', 'init_kernel TEST')
km <- sg('get_kernel_matrix')


# Comm Ulong String
print('CommUlongString')

dump <- sg('send_command', 'add_preproc SORTULONGSTRING')
dump <- sg('set_features', 'TRAIN', traindat_dna, 'DNA')
dump <- sg('send_command', paste('convert TRAIN STRING CHAR STRING ULONG', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TRAIN')

dump <- sg('set_features', 'TEST', testdat_dna, 'DNA')
dump <- sg('send_command', paste('convert TEST STRING CHAR STRING ULONG', order, order-1, gap, reverse))
dump <- sg('send_command', 'attach_preproc TEST')

dump <- sg('send_command', paste('set_kernel COMMSTRING ULONG', size_cache, use_sign, normalization))
dump <- sg('send_command', 'init_kernel TRAIN')
km <- sg('get_kernel_matrix')

dump <- sg('send_command', 'init_kernel TEST')
km <- sg('get_kernel_matrix')


