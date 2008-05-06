# Explicit examples on how to use clustering
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

len <- 12
num <- 30
size_cache <- 10

traindat <- matrix(c(rnorm(len*num)-1,rnorm(len*num)+1), len, 2*num)
trainlab <- c(rep(-1,num),rep(1,num))


# KMEANS
print('KMeans')

k <- 3
iter <- 1000

dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('set_labels', 'TRAIN', trainlab)
dump <- sg('init_distance', 'TRAIN')
dump <- sg('new_classifier', 'KMEANS')
dump <- sg('train_classifier', k, iter)

result <- sg('get_classifier')
radi <- result[[1]]
centers <- result[[2]]


# Hierarchical
print('Hierarchical')

merges=3

dump <- sg('set_features', 'TRAIN', traindat)
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('init_distance', 'TRAIN')
dump <- sg('new_classifier', 'HIERARCHICAL')
dump <- sg('train_classifier', merges)

result <- sg('get_classifier')
merge_distances <- result[[1]]
pairs <- result[[2]]
