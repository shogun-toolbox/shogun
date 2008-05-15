# this is a command file for shogun
    # indented comment
		# tabbed comment

# show help
help

# enable debugging
# loglevel ALL

# execute system command
! ls -l

# example with real valued features
set_features TRAIN data/fm_real.dat
data/fm_real2.dat = get_features TRAIN 

set_kernel GAUSSIAN REAL 10 100.2
init_kernel TRAIN
km.txt=get_kernel_matrix

# example with sparse real valued features
set_features TRAIN data/fm_real_sparse.dat
data/fm_real_sparse2.dat = get_features TRAIN 

set_kernel GAUSSIAN SPARSEREAL 100 1.2
init_kernel TRAIN
km_sparse.txt = get_kernel_matrix

# example with sparse real valued features
set_features TRAIN data/fm_stringchar_dna.dat DNA
data/fm_stringchar_dna2.dat = get_features TRAIN 

set_features TRAIN data/fm_stringchar_dnamatrix.dat DNA
data/fm_stringchar_dnamatrix2.dat = get_features TRAIN 
set_kernel WEIGHTEDDEGREE CHAR 10 20
init_kernel TRAIN
km_dna.txt = get_kernel_matrix
