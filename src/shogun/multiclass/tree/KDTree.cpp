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

#include <shogun/multiclass/tree/KDTree.h>

using namespace shogun;

CKDTree::CKDTree(int32_t leaf_size, EDistanceMetric d)
: CTreeMachine<KDTreeNodeData>()
{
	init();

	m_leaf_size=leaf_size;
	m_dist=d;
}

CKDTree::~CKDTree()
{
}

void CKDTree::build_tree(CDenseFeatures<float64_t>* data)
{
	REQUIRE(data,"data not set\n");
	REQUIRE(m_leaf_size>0,"Leaf size should be greater than 0\n");

	knn_done=false;
	m_data=data->get_feature_matrix();

	vec_id=SGVector<index_t>(m_data.num_cols);
	vec_id.range_fill(0);

	set_root(recursive_build(0,m_data.num_cols-1));
}

void CKDTree::query_knn(CDenseFeatures<float64_t>* data, int32_t k)
{
	REQUIRE(data,"Query data not supplied\n")
	REQUIRE(data->get_num_features()==m_data.num_rows,"query data dimension should be same as training data dimension\n")

	knn_done=true;
	SGMatrix<float64_t> qfeats=data->get_feature_matrix();
	knn_dists=SGMatrix<float64_t>(k,qfeats.num_cols);
	knn_indices=SGMatrix<index_t>(k,qfeats.num_cols);
	int32_t dim=qfeats.num_rows;

	for (int32_t i=0;i<qfeats.num_cols;i++)
	{
		CKNNHeap* heap=new CKNNHeap(k);
		bnode_t* root=NULL;
		if (m_root)
			root=dynamic_cast<bnode_t*>(m_root);

		float64_t min_dist=min_distsq(root,qfeats.matrix+i*dim,dim);
		query_knn_single(heap,min_dist,root,qfeats.matrix+i*dim,dim);
		memcpy(knn_dists.matrix+i*k,heap->get_dists(),k*sizeof(float64_t));
		memcpy(knn_indices.matrix+i*k,heap->get_indices(),k*sizeof(index_t));		

		delete(heap);
	}

	actual_dists(knn_dists);
}

SGMatrix<float64_t> CKDTree::get_knn_dists()
{
	if (knn_done)
		return knn_dists;

	SG_ERROR("knn query has not been executed yet\n");
	return SGMatrix<float64_t>();
}

SGMatrix<index_t> CKDTree::get_knn_indices()
{
	if (knn_done)
		return knn_indices;

	SG_ERROR("knn query has not been executed yet\n");
	return SGMatrix<index_t>();
}

void CKDTree::actual_dists(SGMatrix<float64_t> dists)
{
	if (m_dist==DM_MANHATTAN)
		return;

	for (int32_t i=0;i<dists.num_rows;i++)
	{
		for (int32_t j=0;j<dists.num_cols;j++)
			dists(i,j)=CMath::sqrt(dists(i,j));
	}
}

void CKDTree::query_knn_single(CKNNHeap* heap, float64_t min_dist, bnode_t* node, float64_t* arr, int32_t dim)
{
	if (min_dist>heap->get_max_dist())
		return;

	if (node->data.is_leaf)
	{
		index_t start=node->data.start_idx;
		index_t end=node->data.end_idx;

		for (int32_t i=start;i<=end;i++)
			heap->push(vec_id[i],distance(vec_id[i],arr,dim));

		return;
	}

	bnode_t* cleft=node->left();
	bnode_t* cright=node->right();

	float64_t min_dist_left=min_distsq(cleft,arr,dim);
	float64_t min_dist_right=min_distsq(cright,arr,dim);

	if (min_dist_left<=min_dist_right)
	{
		query_knn_single(heap,min_dist_left,cleft,arr,dim);
		query_knn_single(heap,min_dist_right,cright,arr,dim);		
	}
	else
	{
		query_knn_single(heap,min_dist_right,cright,arr,dim);		
		query_knn_single(heap,min_dist_left,cleft,arr,dim);		
	}

	SG_UNREF(cleft);
	SG_UNREF(cright);
}

float64_t CKDTree::distance(index_t vec, float64_t* arr, int32_t dim)
{
	float64_t ret=0;
	for (int32_t i=0;i<dim;i++)
		ret+=add_dim_dist(m_data(i,vec)-arr[i]);

	return ret;
}

float64_t CKDTree::min_distsq(bnode_t* node,float64_t* feat, int32_t dim)
{
	float64_t dist=0;
	for (int32_t i=0;i<dim;i++)
	{
		float64_t dim_dist=(node->data.bbox_lower[i]-feat[i])+CMath::abs(feat[i]-node->data.bbox_lower[i]);
		dim_dist+=(feat[i]-node->data.bbox_upper[i])+CMath::abs(feat[i]-node->data.bbox_upper[i]);		
		dist+=add_dim_dist(0.5*dim_dist);
	}

	return dist;
}

inline float64_t CKDTree::add_dim_dist(float64_t d)
{
	if (m_dist==DM_EUCLID)
		return d*d;
	else if (m_dist==DM_MANHATTAN)
		return CMath::abs(d);
	else
		SG_ERROR("distance metric not recognized\n");

	return 0;
}

CBinaryTreeMachineNode<KDTreeNodeData>* CKDTree::recursive_build(index_t start, index_t end)
{
	bnode_t* node=new bnode_t();
	init_node(node,start,end);

	// stopping critertia
	if (end-start+1<m_leaf_size*2)
	{
		node->data.is_leaf=true;
		return node;
	}

	node->data.is_leaf=false;
	index_t dim=find_split_dim(node);
	index_t mid=(end+start)/2;
	partition(dim,start,end,mid);

	bnode_t* child_left=recursive_build(start,mid);
	bnode_t* child_right=recursive_build(mid+1,end);

	node->left(child_left);
	node->right(child_right);

	return node;
}

void CKDTree::partition(index_t dim, index_t start, index_t end, index_t mid)
{
	// in-place partial quick-sort
	index_t left=start;
	index_t right=end;
	while (true)
	{
		index_t midindex=left;
		for (int32_t i=left;i<right;i++)
		{
			if (m_data(dim,vec_id[i])<m_data(dim,vec_id[right]))
			{
				CMath::swap(*(vec_id.vector+i),*(vec_id.vector+midindex));
				midindex+=1;
			}
		}

		CMath::swap(*(vec_id.vector+midindex),*(vec_id.vector+right));
		if (midindex==mid)
			break;
		else if (midindex<mid)
			left=midindex+1;
		else
			right=midindex-1;
	}
}

index_t CKDTree::find_split_dim(bnode_t* node)
{
	SGVector<float64_t> upper_bounds=node->data.bbox_upper;
	SGVector<float64_t> lower_bounds=node->data.bbox_lower;

	index_t max_dim=0;
	float64_t max_spread=-1;
	for (int32_t i=0;i<m_data.num_rows;i++)
	{
		float64_t spread=upper_bounds[i]-lower_bounds[i];
		if (spread>max_spread)
		{
			max_spread=spread;
			max_dim=i;
		}
	}

	return max_dim;
}

void CKDTree::init_node(bnode_t* node, index_t start, index_t end)
{
	SGVector<float64_t> upper_bounds(m_data.num_rows);
	SGVector<float64_t> lower_bounds(m_data.num_rows);

	for (int32_t i=0;i<m_data.num_rows;i++)
	{
		upper_bounds[i]=m_data(i,vec_id[start]);
		lower_bounds[i]=m_data(i,vec_id[start]);
		for (int32_t j=start+1;j<=end;j++)
		{
			upper_bounds[i]=CMath::max(upper_bounds[i],m_data(i,vec_id[j]));
			lower_bounds[i]=CMath::min(lower_bounds[i],m_data(i,vec_id[j]));
		}
	}

	node->data.bbox_upper=upper_bounds;
	node->data.bbox_lower=lower_bounds;
	node->data.start_idx=start;
	node->data.end_idx=end;
}

void CKDTree::init()
{
	m_data=SGMatrix<float64_t>();
	m_leaf_size=1;
	vec_id=SGVector<index_t>();
	m_dist=DM_EUCLID;
	knn_done=false;
	knn_dists=SGMatrix<float64_t>();
	knn_indices=SGMatrix<index_t>();	

	SG_ADD(&m_data,"m_data","data matrix",MS_NOT_AVAILABLE);
	SG_ADD(&m_leaf_size,"m_leaf_size","leaf size",MS_NOT_AVAILABLE);
	SG_ADD(&vec_id,"vec_id","id of vectors",MS_NOT_AVAILABLE);
	SG_ADD(&knn_done,"knn_done","knn done or not",MS_NOT_AVAILABLE);
	SG_ADD(&knn_dists,"knn_dists","knn distances",MS_NOT_AVAILABLE);
	SG_ADD(&knn_indices,"knn_indices","knn indices",MS_NOT_AVAILABLE);
}