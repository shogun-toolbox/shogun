from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
parameter_list=[[traindat,10,3,1000],[traindat,11,4,1500]]

def clustering_kmeans (fm_train=traindat, size_cache=10,k=3,iter=1000):
	sg('set_features', 'TRAIN', fm_train)
	sg('set_distance', 'EUCLIDEAN', 'REAL')
	sg('new_clustering', 'KMEANS')
	sg('train_clustering', k, iter)

	[radi, centers]=sg('get_clustering')
	return [radi, centers]

if __name__=='__main__':
	print('KMeans')
	clustering_kmeans(*parameter_list[0])
