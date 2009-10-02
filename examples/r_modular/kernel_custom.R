library(shogun)

# Explicit examples on how to use the different kernels

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))
fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.real(as.matrix(read.table('../data/label_train_dna42.dat')))
fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))
fm_test_cube <- as.matrix(read.table('../data/fm_test_cube.dat', colClasses=c('character')))

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
