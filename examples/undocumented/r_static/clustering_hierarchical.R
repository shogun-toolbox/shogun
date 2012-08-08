library("sg")

fm_train <- t(as.matrix(read.table('../data/fm_train_real.dat')))

# Hierarchical
print('Hierarchical')

merges=3

dump <- sg('set_features', 'TRAIN', fm_train)
dump <- sg('set_distance', 'EUCLIDEAN', 'REAL')
dump <- sg('new_clustering', 'HIERARCHICAL')
dump <- sg('train_clustering', merges)

result <- sg('get_clustering')
merge_distances <- result[[1]]
pairs <- result[[2]]
