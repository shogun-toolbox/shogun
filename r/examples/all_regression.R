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
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)
dump <- sg('send_command', 'new_svm SVRLIGHT')
dump <- sg('send_command', paste('svr_tube_epsilon', tube_epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


# LibSVR
print('LibSVR')

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)
dump <- sg('send_command', 'new_svm LIBSVR')
dump <- sg('send_command', paste('svr_tube_epsilon', tube_epsilon))
dump <- sg('send_command', paste('c', C))
dump <- sg('send_command', 'svm_train')

dump <- sg('set_features', 'TEST', testdat)
dump <- sg('send_command', 'init_kernel TEST')
result <- sg('svm_classify')


#
# misc
#

# KRR
print('KRR')

tau <- 1e-6

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('send_command', paste('set_kernel GAUSSIAN REAL', size_cache, width))
dump <- sg('send_command', 'init_kernel TRAIN')

dump <- sg('set_labels', 'TRAIN', trainlab)

#sg('send_command', 'new_svm KRR')
#sg('send_command', 'set_tau %f' % tau)
#sg('send_command', 'c %f' % C)
#sg('send_command', 'svm_train')

#sg('set_features', 'TEST', testdat)
#sg('send_command', 'init_kernel TEST')
#result <- sg('svm_classify')


