# Explicit examples on how to use clustering
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

fm_train <- as.matrix(read.table('../data/fm_train_real.dat'))


# KMEANS
print('KMeans')

k <- 3
iter <- 1000

dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('set_features', 'TRAIN', fm_train)
dump <- sg('new_clustering', 'KMEANS')
dump <- sg('train_clustering', k, iter)

result <- sg('get_clustering')
radi <- result[[1]]
centers <- result[[2]]
