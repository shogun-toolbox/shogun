/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */


#ifndef _NBODYTREE_H__
#define _NBODYTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/NbodyTreeNodeData.h>
#include <shogun/multiclass/tree/KNNHeap.h> 
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
enum EDistanceMetric
{
	DM_EUCLID,
	DM_MANHATTAN
};

/** @brief This class implements genaralized tree for N-body problems ike k-NN.
 */
class CNbodyTree : public CTreeMachine<NbodyTreeNodeData>
{
public:

	/** constructor
	 * 
	 * @param data data points using which KD-Tree will be made	 
	 * @param leaf_size min number of samples in any node
	 */
	CNbodyTree(int32_t leaf_size=DM_MANHATTAN, EDistanceMetric d=DM_EUCLID);

	/** Destructor */
	virtual ~CNbodyTree() { };

	/** get name
	 * @return class of the tree 
	 */
	virtual const char* get_name() const { return "NbodyTree"; }	

	/** build tree
	 *
	 * @param data data for tree formation
	 */
	void build_tree(CDenseFeatures<float64_t>* data);

	/** apply knn
	 * 
	 * @param data vectors whose KNNs are required
	 * @param k K value in KNN
	 */
	void query_knn(CDenseFeatures<float64_t>* data, int32_t k);

	/** distance b/w KNN vectors and query vectors
	 *
	 * @return distances
	 */
	SGMatrix<float64_t> get_knn_dists();

	/** indices of KNN vectors to query vectors
	 *
	 * @return Matrix of indices
	 */
	SGMatrix<index_t> get_knn_indices();

protected:
	/** find squared minimum distance between node and a query vector
	 * 
	 * @param node present node
	 * @param feat query vector
	 * @param dim dimensions of query vector
	 * @return squared min distance
	 */
	virtual float64_t min_distsq(bnode_t* node,float64_t* feat, int32_t dim)=0;

	/** initialize node
	 *
	 * @param node node to be initialized
	 * @param start start index of index vector
	 * @param end end index of index vector
	 */
	virtual void init_node(bnode_t* node, index_t start, index_t end)=0;

	/** convert squared distances to actual distances
	 *
	 * @param array of distance values
	 * @param len length of array
	 */
	void actual_dists(float64_t* dists, int32_t len);

	/** distance between 2 vectors
	 *
	 * @param index of training data vector
	 * @param arr query vector
	 * @param dim dimension of query vector
	 * @return distance b/w vectors
	 */
	float64_t distance(index_t vec, float64_t* arr, int32_t dim);

	/** compute distance component contributed by present dimension
	 * 
	 * @param d displacement component at chosen dimension
	 * @return distance component
	 */
	inline float64_t add_dim_dist(float64_t d)
	{
		if (m_dist==DM_EUCLID)
			return d*d;
		else if (m_dist==DM_MANHATTAN)
			return CMath::abs(d);
		else
			SG_ERROR("distance metric not recognized\n");

		return 0;
	}	

private:

	/** apply knn on each query vector
	 *
	 * @param heap heap to store kNN distances and indices of corresponding vectors
	 * @param min_dist minimum distance b/ query point and the current node
	 * @param node current node
	 * @param arr current query vector
	 * @param dim dimension of query vector
	 */
	void query_knn_single(CKNNHeap* heap, float64_t min_dist, bnode_t* node, float64_t* arr, int32_t dim);

	/** recursive build
	 * 
	 * @param start start index of index vector for building subtree
	 * @param end index of index vector for building subtree
	 * @return root of subtree built
	 */
	CBinaryTreeMachineNode<NbodyTreeNodeData>* recursive_build(index_t start, index_t end);

	/** rearrange vec_idx between start and end to enable partitioning
	 *
	 * @param dim the chosen dimension of split
	 * @param start start index of index vector
	 * @param end end index of index vector
	 * @param mid the mid index about which split is to be carried out
	 */
	void partition(index_t dim, index_t start, index_t end, index_t mid);

	/** find dim with max spread for split
	 *
	 * @param node node which is to be split
	 * @return split dimension
	 */
	index_t find_split_dim(bnode_t* node);

	/** initialize parameters */
	void init();		

protected:
	/** data matrix */
	SGMatrix<float64_t> m_data;

	/** vector id */
	SGVector<index_t> vec_id;

private:
	/** leaf size */
	int32_t m_leaf_size;

	/** distance metric */
	EDistanceMetric m_dist;

	/** knn query done or not */
	bool knn_done;

	/** knn distances */
	SGMatrix<float64_t> knn_dists;

	/** knn indices */
	SGMatrix<index_t> knn_indices;
};
} /* namespace shogun */

#endif /* _NBODYTREE_H__ */