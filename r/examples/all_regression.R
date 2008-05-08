# Explicit examples on how to use regressions
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30
size_cache <- 10
C <- 10
tube_epsilon <- 1e-2
width <- 2.1

traindat <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
testdat <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
trainlab <- c(rep(-1,num),rep(1,num))

#
# SVM-based
#

# SVR Light
print('SVRLight')

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)
dump <- sg('new_regression', 'SVRLIGHT')
dump <- sg('svr_tube_epsilon', tube_epsilon)
dump <- sg('c', C)
dump <- sg('train_regression')

dump <- sg('set_features', 'TEST', testdat)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


# LibSVR
print('LibSVR')

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)
dump <- sg('new_regression', 'LIBSVR')
dump <- sg('svr_tube_epsilon', tube_epsilon)
dump <- sg('c', C)
dump <- sg('train_regression')

dump <- sg('set_features', 'TEST', testdat)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


#
# misc
#

# KRR
print('KRR')

tau <- 1e-6

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
dump <- sg('init_kernel', 'TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)

dump <- sg('new_regression', 'KRR')
dump <- sg('krr_tau', tau)
dump <- sg('c', C)
dump <- sg('train_regression')

dump <- sg('set_features', 'TEST', testdat)
dump <- sg('init_kernel', 'TEST')
result <- sg('classify')


