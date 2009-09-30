# Explicit examples on how to use clustering
#
# run as R --no-save --slave --file=<filename>

library("sg")
#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

fm_train <- as.matrix(read.table('../data/fm_train_real.dat'))

# Hierarchical
print('Hierarchical')

merges=3

dump <- sg('set_features', 'TRAIN', fm_train)
dump <- sg('set_distance', 'EUCLIDIAN', 'REAL')
dump <- sg('new_clustering', 'HIERARCHICAL')
dump <- sg('train_clustering', merges)

result <- sg('get_clustering')
merge_distances <- result[[1]]
pairs <- result[[2]]
