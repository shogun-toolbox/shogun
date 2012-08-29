library(sg)

source("util/read_mfile.R")
filename <- commandArgs()[6]
indata <- read_mfile(filename)
for (i in 1:length(indata)) {
	eval(parse(text=indata[i]))
}

# there is some randomness involved, alas this is
# not working correctly in R
dump <- sg('init_random', init_random)
set.seed(init_random)

path=strsplit(filename, '/', fixed=TRUE)
module=path[[1]][3]

source(paste(module, '.R', sep=''))
res <- eval(parse(text=paste(module, '()', sep='')))

# res==TRUE is 1; have to negate to get shell-compatible truth value
quit(status=!res)
