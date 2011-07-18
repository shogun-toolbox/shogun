library("sg")

order <- 3
gap <- 0
reverse <- 'n'

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))

# HMM
print('HMM')

N <- 3
M <- 6
order <- 1
hmms <- c()
liks <- c()

dump <- sg('set_features', 'TRAIN', fm_train_cube, 'CUBE')
dump <- sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order)

dump <- sg('new_hmm', N, M)
dump <- sg('bw')
hmm <- sg('get_hmm')

dump <- sg('new_hmm', N, M)
dump <- sg('set_hmm', hmm[[1]], hmm[[2]], hmm[[3]], hmm[[4]])
likelihood <- sg('hmm_likelihood')
