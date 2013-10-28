library("sg")

size_cache <- 10
C <- 10
tube_epsilon <- 1e-2
width <- 2.1

fm_train <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test <- as.matrix(read.table('../data/fm_test_real.dat'))
label_train <- as.double(as.matrix(read.table('../data/label_train_twoclass.dat')))

# SVR Light
dosvrlight <- function()
{
	print('SVRLight')

	dump <- sg('set_features', 'TRAIN', fm_train)
	dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	dump <- sg('set_labels', 'TRAIN', label_train)

	dump <- sg('new_regression', 'SVRLIGHT')
	dump <- sg('svr_tube_epsilon', tube_epsilon)
	dump <- sg('c', C)
	dump <- sg('train_regression')

	dump <- sg('set_features', 'TEST', fm_test)
	result <- sg('classify')
}
try(dosvrlight())
