require(graphics)
library('sg')

#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

traindat <- matrix(1:1000+runif(1000),1,1000)/10
trainlab <- sin(traindat)
testdat <- (matrix(1:1000,1,1000)-1+runif(1000))/10
testlab <- testdat

sg('loglevel', 'ALL')
sg("set_features", "TRAIN", traindat)
sg("set_labels", "TRAIN", trainlab)
sg('set_kernel', 'GAUSSIAN', 'REAL', 50, 20)
sg('new_regression', 'LIBSVR')
sg('c', 0.1)
sg('train_regression')
sg('set_features', 'TEST', testdat)
sg('set_labels', 'TEST', testlab)
out <- sg('classify')
plot(traindat,trainlab, type = 'o', pch='x', col='red');
matplot(testdat,t(matrix(out)), type = 'o', pch='o', col='black',add=T)
