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
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h> 
#include <shogun/multiclass/tree/NbodyTreeNodeData.h>
#include <shogun/multiclass/tree/KNNHeap.h> 
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief This class implements genaralized tree for N-body problems like k-NN, kernel density estimation, 2 point 
 * correlation.
 */
class CNbodyTree : public CTreeMachine<NbodyTreeNodeData>
{
public:

	/** constructor
	 *  
	 * @param leaf_size min number of samples in any node
	 * @param d distance metric to be used	 
	 */
	CNbodyTree(int32_t leaf_size=1, EDistanceType d=D_EUCLIDEAN);

	/** Destructor */
	virtual ~CNbodyTree() { };

	/** get name
	 * @return class of the tree 
	 */
	virtual const char* get_name() const { return "NbodyTree"; }	

	/** get final rearranged vector indices
	 * @return vector indices rearranged corresponding to the built tree
	 */
	SGVector<index_t> get_rearranged_vector_ids() const { return vec_id; }

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

	/** get log of kernel density at query points
	 * 
	 * @param test query points at which kernel density is to be calculated
	 * @param kernel kernel type
	 * @param h width of kernel
	 * @param atol absolute tolerance
	 * @param rtol relative tolerance
	 * @return log kernel density
	 */
	SGVector<float64_t> log_kernel_density(SGMatrix<float64_t> test, EKernelType kernel, float64_t h, float64_t atol, float64_t rtol);

	/** get log of kernel density at query points
	 * 
	 * @param test query points at which kernel density is to be calculated
	 * @param qid id vector of the query tree 
	 * @param qroot root of the query tree
	 * @param kernel kernel type
	 * @param h width of kernel
	 * @param atol absolute tolerance
	 * @param rtol relative tolerance
	 * @return log kernel density
	 */
	SGVector<float64_t> log_kernel_density_dual(SGMatrix<float64_t> test, SGVector<index_t> qid, bnode_t* qroot, EKernelType kernel, float64_t h, float64_t atol, float64_t rtol);

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
	/** find minimum distance between node and a query vector
	 * 
	 * @param node present node
	 * @param feat query vector
	 * @param dim dimensions of query vector
	 * @return min distance
	 */
	virtual float64_t min_dist(bnode_t* node,float64_t* feat, int32_t dim)=0;

	/** find minimum distance between 2 nodes
	 * 
	 * @param nodeq node containing active query vectors
	 * @param noder node containing active training vectors
	 * @return min distance between 2 nodes
	 */
	virtual float64_t min_dist_dual(bnode_t* nodeq, bnode_t* noder)=0;

	/** find max distance between 2 nodes
	 * 
	 * @param nodeq node containing active query vectors
	 * @param noder node containing active training vectors
	 * @return max distance between 2 nodes
	 */
	virtual float64_t max_dist_dual(bnode_t* nodeq, bnode_t* noder)=0;

	/** initialize node
	 *
	 * @param node node to be initialized
	 * @param start start index of index vector
	 * @param end end index of index vector
	 */
	virtual void init_node(bnode_t* node, index_t start, index_t end)=0;

	/** get min as well as max distance of a node from a point
	 *
	 * @param pt point whose distance is to be calculated
	 * @param node node from which distances are to be calculated
	 * @param lower lower bound of distance
	 * @param upper upper bound of distance
	 * @param dim dimension of point vector
	 */
	virtual void min_max_dist(float64_t* pt, bnode_t* node, float64_t &lower,float64_t &upper, int32_t dim)=0;	

	/** convert squared distances to actual distances
	 *
	 * @param dist distance value
	 * @return actual distance
	 */
	inline float64_t actual_dists(float64_t dists)
	{
		if (m_dist==D_MANHATTAN)
			return dists;

		return CMath::sqrt(dists);
	}

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
		if (m_dist==D_EUCLIDEAN)
			return d*d;
		else if (m_dist==D_MANHATTAN)
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

	/** find kde at each query point
	 *
	 * @param node current node
	 * @param data query point
	 * @param kernel kernel type
	 * @param h kernel bandwidth
	 * @param log_atol log of absolute tolerance
	 * @param log_rtol log of relative tolerance
	 * @param log_norm log of kernel norm
	 * @param min_bound_node min evaluated kernel in node
	 * @param spread_node spread of kernel values in node
	 * @param min_bound_global stores the globally calculated min kernel density at query point
	 * @param spread_global spread of kernel values accross entire tree
	 */
	void get_kde_single(bnode_t* node,float64_t* data, EKernelType kernel, float64_t h, float64_t log_atol, float64_t log_rtol,
	float64_t log_norm, float64_t min_bound_node, float64_t spread_node, float64_t &min_bound_global, float64_t &spread_global);

	/** depth-first traversal in dual trees for KDE
	 *
	 * @param refnode current node from reference tree
	 * @param querynode current node from query tree
	 * @param qid id vector of query tree
	 * @param qdata query data matrix
	 * @param log_density stores log of kernel density at each query point
	 * @param kernel_type kernel type used
	 * @param h kernel bandwidth
	 * @param log_atol log absolute tolerance
	 * @param log_rtol log relative tolerance
	 * @param log_norm log of kernel norm
	 * @param min_bound_node min evaluated kernel in node
	 * @param spread_node spread of kernel values in node
	 * @param min_bound_global stores the globally calculated min kernel density for all query points
	 * @param spread_global spread of kernel values accross entire reference tree for all query points in query tree
	 */
	void kde_dual(bnode_t* refnode, bnode_t* querynode, SGVector<index_t> qid, SGMatrix<float64_t> qdata, SGVector<float64_t> log_density, 
	EKernelType kernel_type, float64_t h, float64_t log_atol, float64_t log_rtol, float64_t log_norm, float64_t min_bound_node, 
	float64_t spread_node, float64_t &min_bound_global, float64_t &spread_global);

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

	/** log-sum-exp trick for 2 numbers 
	 * 
	 * @param x number 1
	 * @param y number 2
	 * @return log of sum of exp of numbers
	 */
	inline float64_t logsumexp(float64_t x, float64_t y)
	{
		float64_t a=CMath::max(x,y);
		if (a==-CMath::INFTY)
			return -CMath::INFTY;
		
		return a+CMath::log(CMath::exp(x-a)+CMath::exp(y-a));
	}

	/** log-diff-exp trick for 2 numbers 
	 * 
	 * @param x number 1
	 * @param y number 2
	 * @return log of difference of exp of numbers
	 */
	inline float64_t logdiffexp(float64_t x, float64_t y)
	{
		if (x<=y)
			return -CMath::INFTY;

		return x+CMath::log(1-CMath::exp(y-x));
	}

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
	EDistanceType m_dist;

	/** knn query done or not */
	bool knn_done;

	/** knn distances */
	SGMatrix<float64_t> knn_dists;

	/** knn indices */
	SGMatrix<index_t> knn_indices;
};
} /* namespace shogun */

#endif /* _NBODYTREE_H__ */