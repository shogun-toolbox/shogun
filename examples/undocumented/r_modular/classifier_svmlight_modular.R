library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(read.table('../data/label_train_dna.dat')$V1)

# svm light
dosvmlight <- function()
{
	print('SVMLight')

	feats_train <- StringCharFeatures("DNA")
	dump <- feats_train$set_features(feats_train, fm_train_dna)
	feats_test <- StringCharFeatures("DNA")
	dump <- feats_test$set_features(feats_test, fm_test_dna)
	degree <- as.integer(20)

	kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C <- 1.017
	epsilon <- 1e-5
	num_threads <- as.integer(3)
	labels <- Labels(as.real(label_train_dna))

	svm <- SVMLight(C, kernel, labels)
	dump <- svm$set_epsilon(svm, epsilon)
	dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
	dump <- svm$train(svm)

	dump <- kernel$init(kernel, feats_train, feats_test)
	lab <- svm$apply(svm)
	out <- lab$get_labels(lab)
}
try(dosvmlight())
