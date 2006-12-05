# The purpose of this file is to supply no functionality
# except easier access functions in R for external C 
# function calls. 
#
# For example instead of typing
#
#     > .External("sg", "send_command", "blah")
#
# one can simply type
#
#     > send_command(blah)
#
# where > is the R prompt.

# interface sg(arg1,arg2,...) as w/ matlab/octave/python
#
sg <- function(...) .External("sg",...,PACKAGE="sg")


# R specific interface

# Generic functions
#
send_command <- function(x) .External("sg","send_command",x,PACKAGE="sg")
set_features <- function(x,y) .External("sg","set_features",x,y,PACKAGE="sg")
add_features <- function(x,y) .External("sg","add_features",x,y,PACKAGE="sg")
set_labels <- function(x,y) .External("sg","set_labels", x,y,PACKAGE="sg")
get_kernel_matrix <- function() .External("sg","get_kernel_matrix",PACKAGE="sg")

# SVM functions
#
svm_classify <- function() .External("sg","svm_classify",PACKAGE="sg")
get_svm <- function() .External("sg","get_svm",PACKAGE="sg")
get_subkernel_weights <- function() .External("sg","get_subkernel_weights",PACKAGE="sg") 

# HMM functions
#
get_hmm <- function() .External("sg","get_hmm",PACKAGE="sg")
