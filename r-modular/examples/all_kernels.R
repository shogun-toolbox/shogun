dyn.load('features/Features.so')
dyn.load('kernel/Kernel.so')
dyn.load('distance/Distance.so')
source('features/Features.R')
source('kernel/Kernel.R')
source('distance/Distance.R')
cacheMetaData(1)

num <- 12
leng <- 50
rep <- 5
weight <- 0.3
dist <- 1.2

# Explicit examples on how to use the different kernels

traindata_real <- matrix(c(rnorm(num*leng)-dist,rnorm(num*leng)+dist),leng,2*num)
testdata_real <- matrix(c(rnorm(num*leng)-dist,rnorm(num*leng)+dist),leng,2*num)

#traindata_byte <- uint8(256*[rand(leng,2*num)])
#testdata_byte <- uint8(256*[rand(leng,3*num)])
#
#traindata_word <- uint16((2^16)*[rand(leng,2*num)])
#testdata_word <- uint16((2^16)*[rand(leng,3*num)])

# generate some random DNA  <- ;-]
acgt <- c('A', 'C', 'G', 'T')
trainlab_dna <- c(rep(1,num/2),rep(-1,num/2))
traindata_dna <- list()
testdata_dna <- list()
for (i in 1:num)
{
	traindata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep <- "", collapse <- "")
	testdata_dna[i] <- paste(acgt[ceiling(4*runif(len))], sep <- "", collapse <- "")
}

# generate a sequence with characters 1-6 drawn from 3 loaded cubes
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

############################################################################
## byte features
############################################################################
#
## linear byte
#print('LinearByte')
#
#num_feats <- 11
#feats_train <- ByteFeatures(RAWBYTE)
#feats_train$copy_feature_matrix(traindata_byte)
#
#feats_test <- ByteFeatures(RAWBYTE)
#feats_test$copy_feature_matrix(testdata_byte)
#
#kernel <- LinearByteKernel(feats_train, feats_train)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
###########################################################################
# real features
###########################################################################

# chi2
print('Chi2')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# const
print('Const')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
c <- 23.

kernel <- ConstKernel(feats_train, feats_train, c)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# diag
print('Diag')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
diag <- 23.

kernel <- DiagKernel(feats_train, feats_train, diag)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# gaussian
print('Gaussian')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 1.9

kernel <- GaussianKernel(feats_train, feats_train, width)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# gaussian_shift
print('GaussianShift')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 1.8
max_shift <- as.integer(2)
shift_step <- as.integer(1)

kernel <- GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# linear
print('Linear')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
scale <- 1.2

kernel <- LinearKernel(feats_train, feats_train, scale)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# poly
print('Poly')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
degree <- as.integer(4)
inhomogene <- FALSE
use_normalization <- TRUE

kernel <- PolyKernel(
	feats_train, feats_train, degree, inhomogene, use_normalization)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sigmoid
print('Sigmoid')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
size_cache <- as.integer(10)
gamma <- 1.2
coef0 <- 1.3

kernel <- SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# sparse real features
###########################################################################

# sparse_gaussian
print('SparseGaussian')

feat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, feat)
width <- 1.1

kernel <- SparseGaussianKernel(feats_train, feats_train, width)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sparse_linear
print('SparseLinear')

feat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, feat)
scale <- 1.1

kernel <- SparseLinearKernel(feats_train, feats_train, scale)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sparse_poly
print('SparsePoly')

feat <- RealFeatures(traindata_real)
feats_train <- SparseRealFeatures()
feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(testdata_real)
feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(feats_test, feat)
size_cache <- as.integer(10)
degree <- as.integer(3)
inhomogene <- TRUE
use_normalization <- FALSE

kernel <- SparsePolyKernel(feats_train, feats_train, size_cache, degree,
	inhomogene, use_normalization)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# word features
###########################################################################
#
## linear_word
#print('LinearWord')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#do_rescale <- TRUE
#scale <- 1.4
#
#kernel <- LinearWordKernel(feats_train, feats_train, do_rescale, scale)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## poly_match_word
#print('PolyMatchWord')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#degree <- 2
#inhomogene <- TRUE
#
#kernel <- PolyMatchWordKernel(feats_train, feats_train, degree, inhomogene)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## word_match
#print('WordMatch')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#degree <- 3
#do_rescale <- TRUE
#scale <- 1.4
#
#kernel <- WordMatchKernel(feats_train, feats_train, degree, do_rescale, scale)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
###########################################################################
# string features
############################################################################
#
## fixed_degree_string
#print('FixedDegreeString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#degree <- 3
#
#kernel <- FixedDegreeStringKernel(feats_train, feats_train, degree)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## linear_string
#print('LinearString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#
#kernel <- LinearStringKernel(feats_train, feats_train)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## local_alignment_strin
#print('LocalAlignmentString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#
#kernel <- LocalAlignmentStringKernel(feats_train, feats_train)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## poly_match_string
#print('PolyMatchString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#degree <- 3
#inhomogene <- FALSE
#
#kernel <- PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## simple_locality_improved_string
#print('SimpleLocalityImprovedString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#l <- 5
#inner_degree <- 5
#outer_degree <- 7
#
#kernel <- SimpleLocalityImprovedStringKernel(
#	feats_train, feats_train, l, inner_degree, outer_degree)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## weighted_degree_string
#print('WeightedDegreeString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#degree <- 20
#
#kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)
#
##weights <- arange(1,degree+1,dtype <- double)[::-1]/ \
##	sum(arange(1,degree+1,dtype <- double))
##kernel.set_wd_weights(weights)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## weighted_degree_position_string
#print('WeightedDegreePositionString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#degree <- 20
#
#kernel <- WeightedDegreePositionStringKernel(feats_train, feats_train, degree)
#
##kernel$set_shifts(zeros(len(traindata_dna[0]), dtype <- int))
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## locality_improved_string
#print('LocalityImprovedString')
#
#feats_train <- StringCharFeatures(DNA)
#feats_train$set_string_features(traindata_dna)
#feats_test <- StringCharFeatures(DNA)
#feats_test$set_string_features(testdata_dna)
#l <- 5
#inner_degree <- 5
#outer_degree <- 7
#
#kernel <- LocalityImprovedStringKernel(
#	feats_train, feats_train, l, inner_degree, outer_degree)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
############################################################################
## complex string features
############################################################################
#
## comm_word_string
#print('CommWordString')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(kernel, feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#use_sign <- FALSE
#normalization <- FULL_NORMALIZATION
#
#kernel <- CommWordStringKernel(
#	feats_train, feats_train, use_sign, normalization)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## weighted_comm_word_string
#print('WeightedCommWordString')
#
#order <- 3
#gap <- 0
#reverse <- TRUE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(kernel, feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#use_sign <- FALSE
#normalization <- FULL_NORMALIZATION
#
#kernel <- WeightedCommWordStringKernel(
#	feats_train, feats_train, use_sign, normalization)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## comm_ulong_string
#print('CommUlongString')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringUlongFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortUlongString()
#preproc$init(kernel, feats_train)
#feats_train$add_preproc(preproc)
#feats_train$apply_preproc()
#
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringUlongFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#feats_test$add_preproc(preproc)
#feats_test$apply_preproc()
#
#use_sign <- FALSE
#normalization <- FULL_NORMALIZATION
#
#kernel <- CommUlongStringKernel(
#	feats_train, feats_train, use_sign, normalization)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
###########################################################################
# misc kernels
###########################################################################

## custom
#print('Custom')
#
#dim <- 7
#data <- rand(dim, dim)
#feats <- RealFeatures(data)
#symdata <- data+data'
#lowertriangle <- array([symdata[(x,y)] for x in xrange(symdata.shape[1])
#	for y in xrange(symdata.shape[0]) if y< <- x])
#
#kernel <- CustomKernel(feats, feats)
#
#kernel$set_triangle_kernel_matrix_from_triangle(lowertriangle)
#km_triangletriangle <- kernel$get_kernel_matrix()
#
#kernel$set_triangle_kernel_matrix_from_full(symdata)
#km_fulltriangle <- kernel$get_kernel_matrix()
#
#kernel$set_full_kernel_matrix_from_full(data)
#km_fullfull <- kernel$get_kernel_matrix()

# distance
print('Distance')

feats_train <- RealFeatures(traindata_real)
feats_test <- RealFeatures(testdata_real)
width <- 1.7
distance <- EuclidianDistance()

kernel <- DistanceKernel(feats_train, feats_test, width, distance)

km_train <- kernel$get_kernel_matrix()
kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# auc
#print('AUC')
#
#feats_train <- RealFeatures(traindata_real)
#feats_test <- RealFeatures(testdata_real)
#width <- 1.7
#subkernel <- GaussianKernel(feats_train, feats_test, width)
#
#num_feats <- 2; # do not change!
#len_train <- 11
#len_test <- 17
#data <- uint16((len_train-1)*rand(num_feats, len_train))
#feats_train <- WordFeatures(data)
#data <- uint16((len_test-1)*rand(num_feats, len_test))
#feats_test <- WordFeatures(data)
#
#kernel <- AUCKernel(feats_train, feats_test, subkernel)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()

## combined
#print('Combined')
#
#kernel <- CombinedKernel()
#feats_train <- CombinedFeatures()
#feats_test <- CombinedFeatures()
#
#subkfeats_train <- StringCharFeatures(DNA)
#subkfeats_train$set_string_features(traindata_dna)
#subkfeats_test <- StringCharFeatures(DNA)
#subkfeats_test$set_string_features(testdata_dna)
#subkernel <- LinearStringKernel(10)
#feats_train$append_feature_obj(subkfeats_train)
#feats_test$append_feature_obj(subkfeats_test)
#kernel$append_kernel(subkernel)
#
#subkfeats_train <- StringCharFeatures(DNA)
#subkfeats_train$set_string_features(traindata_dna)
#subkfeats_test <- StringCharFeatures(DNA)
#subkfeats_test$set_string_features(testdata_dna)
#degree <- 3
#subkernel <- FixedDegreeStringKernel(10, degree)
#feats_train$append_feature_obj(subkfeats_train)
#feats_test$append_feature_obj(subkfeats_test)
#kernel$append_kernel(subkernel)
#
#subkfeats_train <- StringCharFeatures(DNA)
#subkfeats_train$set_string_features(traindata_dna)
#subkfeats_test <- StringCharFeatures(DNA)
#subkfeats_test$set_string_features(testdata_dna)
#subkernel <- LocalAlignmentStringKernel(10)
#feats_train$append_feature_obj(subkfeats_train)
#feats_test$append_feature_obj(subkfeats_test)
#kernel$append_kernel(subkernel)
#
#kernel$init(kernel, feats_train, feats_train)
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
## plugin_estimate
#print('PluginEstimate w/ HistogramWord')
#
#order <- 3
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(traindata_dna)
#feats_train <- StringWordFeatures(charfeat$get_alphabet())
#feats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#
#charfeat <- StringCharFeatures(DNA)
#charfeat$set_string_features(testdata_dna)
#feats_test <- StringWordFeatures(charfeat$get_alphabet())
#feats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#
#pie <- PluginEstimate()
#lab <- round(rand(1, feats_train$get_num_vectors()))*2-1
#labels <- Labels(lab)
#pie$set_labels(labels)
#pie$set_features(feats_train)
#pie$train()
#
#kernel <- HistogramWordKernel(feats_train, feats_train, pie)
#km_train <- kernel$get_kernel_matrix()
#
#kernel$init(kernel, feats_train, feats_test)
#pie$set_features(feats_test)
#pie$classify()$get_labels()
#km_test <- kernel$get_kernel_matrix()
#
## top_fisher
#print('TOP/Fisher on PolyKernel')
#
#N <- 3
#M <- 6
#pseudo <- 1e-1
#order <- 1
#gap <- 0
#reverse <- FALSE
#
#charfeat <- StringCharFeatures(CUBE)
#charfeat$set_string_features(cubesequence)
#wordfeats_train <- StringWordFeatures(charfeat$get_alphabet())
#wordfeats_train$obtain_from_char(charfeat, order-1, order, gap, reverse)
#preproc <- SortWordString()
#preproc$init(kernel, wordfeats_train)
#wordfeats_train$add_preproc(preproc)
#wordfeats_train$apply_preproc()
#
#charfeat <- StringCharFeatures(CUBE)
#charfeat$set_string_features(cubesequence)
#wordfeats_test <- StringWordFeatures(charfeat$get_alphabet())
#wordfeats_test$obtain_from_char(charfeat, order-1, order, gap, reverse)
#wordfeats_test$add_preproc(preproc)
#wordfeats_test$apply_preproc()
#
#pos <- HMM(wordfeats_train, N, M, pseudo)
#pos$train()
#pos$baum_welch_viterbi_train(BW_NORMAL)
#neg <- HMM(wordfeats_train, N, M, pseudo)
#neg$train()
#neg$baum_welch_viterbi_train(BW_NORMAL)
#pos_clone <- HMM(pos)
#neg_clone <- HMM(neg)
#pos_clone$set_observations(wordfeats_test)
#neg_clone$set_observations(wordfeats_test)
#
#feats_train <- TOPFeatures(10, pos, neg, FALSE, FALSE)
#feats_test <- TOPFeatures(10, pos_clone, neg_clone, FALSE, FALSE)
#kernel <- PolyKernel(feats_train, feats_train, 1, FALSE, TRUE)
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
#
#feats_train <- FKFeatures(10, pos, neg)
#feats_train$set_opt_a(-1); #estimate prior
#feats_test <- FKFeatures(10, pos_clone, neg_clone)
#feats_test$set_a(feats_train$get_a()); #use prior from training data
#kernel <- PolyKernel(feats_train, feats_train, 1, FALSE, TRUE)
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
