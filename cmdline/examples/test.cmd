# this is a command file for shogun
    # indented comment
		# tabbed comment

loglevel ALL
! ls -l



#a=bla

set_features TRAIN y
set_kernel GAUSSIAN REAL 10 1.2
init_kernel TRAIN

set_features TRAIN z
set_kernel GAUSSIAN SPARSEREAL 10 1.2
init_kernel TRAIN

km.txt = get_kernel_matrix
