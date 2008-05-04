dyn.load('features/Features.so')
dyn.load('preproc/PreProc.so')
dyn.load('distributions/Distribution.so')
source('preproc/PreProc.R')
source('features/Features.R')
source('distributions/Distribution.R')
cacheMetaData(1)

num=12
leng=50
rep=5
weight=0.3

# Explicit examples on how to use distributions

# generate some random DNA =;-]
#acgt='ACGT'
acgt <- c('A', 'C', 'G', 'T')
trainlab_dna=c(rep(1,num/2),rep(-1,num/2))
traindata_dna=list()
testdata_dna=list()
for (i in 1:num)
{
	traindata_dna[i]=paste(acgt[ceiling(4*runif(leng))], sep="", collapse="")
	testdata_dna[i]=paste(acgt[ceiling(4*runif(leng))], sep="", collapse="")
}

traindata_dna=c(traindata_dna,recursive=TRUE)
testdata_dna=c(testdata_dna,recursive=TRUE)

cube <- list(NULL, NULL, NULL)
numrep <- vector(mode='numeric',length=18)+100
numrep[1] <- 0;
numrep[2] <- 0;
numrep[3] <- 0;
numrep[10] <- 0;
numrep[11] <- 0;
numrep[12] <- 0;

for (c in 1:3)
{
	for (i in 1:6)
	{
		cube[[c]] <- c(cube[[c]], vector(mode='numeric',length=numrep[(c-1)*6+i])+i)
	}
	cube[[c]] <- sample(cube[[c]],300,replace=TRUE);
}

cube <- c(cube[[1]], cube[[2]], cube[[3]])
cubesequence <- paste(cube, sep="", collapse="")

###########################################################################
# distributions
###########################################################################

# Histogram
print('Histogram')

order=as.integer(3)
start=as.integer(order-1)
gap=as.integer(0)
reverse=FALSE

charfeat=StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, traindata_dna)
feats=StringWordFeatures(charfeat$get_alphabet())
feats$obtain_from_char(feats, charfeat, start, order, gap, reverse)
preproc=SortWordString()
preproc$init(preproc, feats)
feats$add_preproc(feats, preproc)
feats$apply_preproc(feats)

histo=Histogram(feats)
histo$train()

histo$get_histogram()

num_examples=feats$get_num_vectors()
num_param=histo$get_num_model_parameters()

# commented out as this is quite time consuming
#derivs=matrix(0,num_param, num_examples)
#for (i in 0:(num_examples-1))
#{
#	for (j in 0:(num_param-1))
#	{
#		derivs[j,i]=histo$get_log_derivative(histo, j, i)
#	}
#}
histo$get_log_likelihood(histo, 0)
histo$get_log_likelihood_sample()

# Linear HMM
print('LinearHMM')

order=3
gap=0
reverse=FALSE

charfeat=StringCharFeatures("DNA")
charfeat$set_string_features(charfeat, traindata_dna)
feats=StringWordFeatures(charfeat$get_alphabet())
feats$obtain_from_char(feats, charfeat, order-1, order, gap, reverse)
preproc=SortWordString()
preproc$init(preproc, feats)
feats$add_preproc(feats, preproc)
feats$apply_preproc(feats)

hmm=LinearHMM(feats)
hmm$train(hmm)

hmm$get_transition_probs()

num_examples=feats$get_num_vectors()
num_param=hmm$get_num_model_parameters()
derivs=matrix(0,num_param, num_examples)

for (i in 0:(num_examples-1))
{
	for (j in 0:(num_param-1))
	{
		derivs[j,i]=hmm$get_log_derivative(hmm, j, i)
	}
}

hmm$get_log_likelihood(hmm, 0)
hmm$get_log_likelihood_sample()

# HMM
print('HMM')

N=3
M=6
pseudo=1e-1
order=1
gap=0
reverse=FALSE
num_examples=2
charfeat=StringCharFeatures("CUBE")
charfeat$set_string_features(charfeat, cubesequence)
feats=StringWordFeatures(charfeat$get_alphabet())
feats$obtain_from_char(feats, charfeat, order-1, order, gap, reverse)
preproc=SortWordString()
preproc$init(preproc, feats)
feats$add_preproc(feats, preproc)
feats$apply_preproc(feats)

hmm=HMM(feats, N, M, pseudo)
hmm$train(hmm)
hmm$baum_welch_viterbi_train(hmm, "BW_NORMAL")

num_examples=feats$get_num_vectors()
num_param=hmm$get_num_model_parameters()

derivs=matrix(0,num_param, num_examples)
for (i in 0:(num_examples-1))
{
	for (j in 0:(num_param-1))
	{
		derivs[j,i]=hmm$get_log_derivative(hmm, j, i)
	}
}

best_path=0
best_path_state=0

for (i in 0:(num_examples-1))
{
	best_path = best_path + hmm$best_path(hmm, i)
	for (j in 0:(N-1))
	{
		best_path_state = best_path_state + hmm$get_best_path_state(hmm, i, j)
	}
}

hmm$get_log_likelihood(hmm, 0)
hmm$get_log_likelihood_sample()
