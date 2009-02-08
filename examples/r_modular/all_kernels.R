library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

weight <- 0.3

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

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# const
print('Const')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
c <- 23.

kernel <- ConstKernel(feats_train, feats_train, c)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# diag
print('Diag')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
diag <- 23.

kernel <- DiagKernel(feats_train, feats_train, diag)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# gaussian
print('Gaussian')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.9

kernel <- GaussianKernel(feats_train, feats_train, width)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# gaussian_shift
print('GaussianShift')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.8
max_shift <- as.integer(2)
shift_step <- as.integer(1)

kernel <- GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# linear
print('Linear')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
scale <- 1.2

kernel <- LinearKernel()
dump <- kernel$set_normalizer(kernel, AvgDiagKernelNormalizer(scale))
dump <- kernel$init(kernel, feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# poly
print('Poly')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
degree <- as.integer(4)
inhomogene <- FALSE

kernel <- PolyKernel(
	feats_train, feats_train, degree, inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sigmoid
print('Sigmoid')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
size_cache <- as.integer(10)
gamma <- 1.2
coef0 <- 1.3

kernel <- SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# sparse real features
###########################################################################

# sparse_gaussian
print('SparseGaussian')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
width <- 1.1

kernel <- SparseGaussianKernel(feats_train, feats_train, width)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sparse_linear
print('SparseLinear')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
scale <- 1.1

kernel <- SparseLinearKernel()
dump <- kernel$set_normalizer(kernel, AvgDiagKernelNormalizer(scale))
dump <- kernel$init(kernel, feats_train, feats_train)


km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# sparse_poly
print('SparsePoly')

feat <- RealFeatures(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(feats_train, feat)
feat <- RealFeatures(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(feats_test, feat)
size_cache <- as.integer(10)
degree <- as.integer(3)
inhomogene <- TRUE

kernel <- SparsePolyKernel(feats_train, feats_train, size_cache, degree,
	inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
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

# fixed_degree_string
print('FixedDegreeString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(3)

kernel <- FixedDegreeStringKernel(feats_train, feats_train, degree)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# linear_string
print('LinearString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)

kernel <- LinearStringKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# local_alignment_strin
print('LocalAlignmentString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)

kernel <- LocalAlignmentStringKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# oligo_string
print('OligoString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
k <- as.integer(3)
width <- 1.2
size_cache <- as.integer(10)

kernel <- OligoKernel(size_cache,  k, width)
dump <- kernel$init(kernel, feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# poly_match_string
print('PolyMatchString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(3)
inhomogene <- FALSE

kernel <- PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# simple_locality_improved_string
print('SimpleLocalityImprovedString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
l <- as.integer(5)
inner_degree <- as.integer(5)
outer_degree <- as.integer(7)

kernel <- SimpleLocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# weighted_degree_string
print('WeightedDegreeString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(20)

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

#weights <- arange(1,degree+1,dtype <- double)[::-1]/ \
#	sum(arange(1,degree+1,dtype <- double))
#kernel$set_wd_weights(weights)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# weighted_degree_position_string
print('WeightedDegreePositionString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
degree <- as.integer(20)

kernel <- WeightedDegreePositionStringKernel(feats_train, feats_train, degree)

#kernel$set_shifts(zeros(len(fm_train_dna[0]), dtype <- int))

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# locality_improved_string
print('LocalityImprovedString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_string_features(feats_train, fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_string_features(feats_test, fm_test_dna)
l <- as.integer(5)
inner_degree <- as.integer(5)
outer_degree <- as.integer(7)

kernel <- LocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

###########################################################################
# complex string features
###########################################################################

# comm_word_string
print('CommWordString')

order <- as.integer(3)
gap <- as.integer(0)
start <- as.integer(order-1)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

kernel <- CommWordStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# weighted_comm_word_string
print('WeightedCommWordString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- TRUE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

kernel <- WeightedCommWordStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# comm_ulong_string
print('CommUlongString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)
preproc <- SortUlongString()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)


charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringUlongFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

use_sign <- FALSE

kernel <- CommUlongStringKernel(feats_train, feats_train, use_sign)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

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

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)
width <- 1.7
distance <- EuclidianDistance()

kernel <- DistanceKernel(feats_train, feats_test, width, distance)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# auc
#print('AUC')
#
#feats_train <- RealFeatures(fm_train_real)
#feats_test <- RealFeatures(fm_test_real)
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

# combined
print('Combined')

kernel <- CombinedKernel()
feats_train <- CombinedFeatures()
feats_test <- CombinedFeatures()

subkfeats_train <- RealFeatures(fm_train_real)
subkfeats_test <- RealFeatures(fm_test_real)
subkernel <- GaussianKernel(as.integer(10), 1.6)
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

subkfeats_train <- StringCharFeatures("DNA")
dump <- subkfeats_train$set_string_features(subkfeats_train, fm_train_dna)
subkfeats_test <- StringCharFeatures("DNA")
dump <- subkfeats_test$set_string_features(subkfeats_test, fm_test_dna)
degree <- as.integer(3)
subkernel <- FixedDegreeStringKernel(as.integer(10), degree)
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

subkfeats_train <- StringCharFeatures("DNA")
dump <- subkfeats_train$set_string_features(subkfeats_train, fm_train_dna)
subkfeats_test <- StringCharFeatures("DNA")
dump <- subkfeats_test$set_string_features(subkfeats_test, fm_test_dna)
subkernel <- LocalAlignmentStringKernel(as.integer(10))
dump <- feats_train$append_feature_obj(feats_train, subkfeats_train)
dump <- feats_test$append_feature_obj(feats_test, subkfeats_test)
dump <- kernel$append_kernel(kernel, subkernel)

dump <- kernel$init(kernel, feats_train, feats_train)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

# plugin_estimate
print('PluginEstimate w/ HistogramWord')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(feats_train, charfeat, start, order, gap, reverse)

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_string_features(charfeat, fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(feats_test, charfeat, start, order, gap, reverse)

pie <- PluginEstimate()
labels <- Labels(label_train_dna)
dump <- pie$set_labels(pie, labels)
dump <- pie$set_features(pie, feats_train)
dump <- pie$train()

kernel <- HistogramWordStringKernel(feats_train, feats_train, pie)
km_train <- kernel$get_kernel_matrix()

dump <- kernel$init(kernel, feats_train, feats_test)
dump <- pie$set_features(pie, feats_test)
km_test <- kernel$get_kernel_matrix()

# top_fisher
print('TOP/Fisher on PolyKernel')

N <- as.integer(3)
M <- as.integer(6)
pseudo <- 1e-1
order <- as.integer(1)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE

charfeat <- StringCharFeatures("CUBE")
dump <- charfeat$set_string_features(charfeat, fm_train_cube)
wordfeats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- wordfeats_train$obtain_from_char(wordfeats_train, charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(preproc, wordfeats_train)
dump <- wordfeats_train$add_preproc(wordfeats_train, preproc)
dump <- wordfeats_train$apply_preproc(wordfeats_train)

charfeat <- StringCharFeatures("CUBE")
dump <- charfeat$set_string_features(charfeat, fm_test_cube)
wordfeats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- wordfeats_test$obtain_from_char(wordfeats_test, charfeat, start, order, gap, reverse)
dump <- wordfeats_test$add_preproc(wordfeats_test, preproc)
dump <- wordfeats_test$apply_preproc(wordfeats_test)

pos <- HMM(wordfeats_train, N, M, pseudo)
dump <- pos$train()
dump <- pos$baum_welch_viterbi_train(pos, "BW_NORMAL")
neg <- HMM(wordfeats_train, N, M, pseudo)
dump <- neg$train()
dump <- neg$baum_welch_viterbi_train(neg, "BW_NORMAL")
pos_clone <- HMM(pos)
neg_clone <- HMM(neg)
dump <- pos_clone$set_observations(pos_clone, wordfeats_test)
dump <- neg_clone$set_observations(neg_clone, wordfeats_test)

feats_train <- TOPFeatures(size_cache, pos, neg, FALSE, FALSE)
feats_test <- TOPFeatures(size_cache, pos_clone, neg_clone, FALSE, FALSE)
kernel <- PolyKernel(feats_train, feats_train, as.integer(1), FALSE)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()

feats_train <- FKFeatures(size_cache, pos, neg)
dump <- feats_train$set_opt_a(feats_train, -1); #estimate prior
feats_test <- FKFeatures(size_cache, pos_clone, neg_clone)
dump <- feats_test$set_a(feats_test, feats_train$get_a()); #use prior from training data
kernel <- PolyKernel(feats_train, feats_train, as.integer(1), FALSE)
km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
