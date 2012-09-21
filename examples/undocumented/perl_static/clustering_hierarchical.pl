from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
parameter_list=[[traindat,10,3],[traindat,11,4]]
def clustering_hierarchical (fm_train=traindat, size_cache=10,merges=3):

	sg('set_features', 'TRAIN', fm_train)
	sg('set_distance', 'EUCLIDEAN', 'REAL')
	sg('new_clustering', 'HIERARCHICAL')
	sg('train_clustering', merges)

	[merge_distance, pairs]=sg('get_clustering')
	return [merge_distance, pairs]

if __name__=='__main__':
	print('Hierarchical')
	clustering_hierarchical(*parameter_list[0])
