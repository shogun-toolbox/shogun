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

#include <vector>

#include <shogun/lib/View.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/tree/CARTree.h>

using namespace Eigen;
using namespace shogun;

const float64_t CCARTree::MISSING=CMath::MAX_REAL_NUMBER;
const float64_t CCARTree::EQ_DELTA=1e-7;
const float64_t CCARTree::MIN_SPLIT_GAIN=1e-7;

CCARTree::CCARTree()
: CTreeMachine<CARTreeNodeData>()
{
	init();
}

CCARTree::CCARTree(SGVector<bool> attribute_types, EProblemType prob_type)
: CTreeMachine<CARTreeNodeData>()
{
	init();
	set_feature_types(attribute_types);
	set_machine_problem_type(prob_type);
}

CCARTree::CCARTree(SGVector<bool> attribute_types, EProblemType prob_type, int32_t num_folds, bool cv_prune)
: CTreeMachine<CARTreeNodeData>()
{
	init();
	set_feature_types(attribute_types);
	set_machine_problem_type(prob_type);
	set_num_folds(num_folds);
	if (cv_prune)
		set_cv_pruning(cv_prune);
}

CCARTree::~CCARTree()
{
	SG_UNREF(m_alphas);
}

void CCARTree::set_labels(CLabels* lab)
{
	if (lab->get_label_type()==LT_MULTICLASS)
		set_machine_problem_type(PT_MULTICLASS);
	else if (lab->get_label_type()==LT_REGRESSION)
		set_machine_problem_type(PT_REGRESSION);
	else
		SG_ERROR("label type supplied is not supported\n")

	SG_REF(lab);
	SG_UNREF(m_labels);
	m_labels=lab;
}

void CCARTree::set_machine_problem_type(EProblemType mode)
{
	m_mode=mode;
}

bool CCARTree::is_label_valid(CLabels* lab) const
{
	if (m_mode==PT_MULTICLASS && lab->get_label_type()==LT_MULTICLASS)
		return true;
	else if (m_mode==PT_REGRESSION && lab->get_label_type()==LT_REGRESSION)
		return true;
	else
		return false;
}

CMulticlassLabels* CCARTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	// apply multiclass starting from root
	bnode_t* current=dynamic_cast<bnode_t*>(get_root());

	REQUIRE(current, "Tree machine not yet trained.\n");
	CLabels* ret=apply_from_current_node(data->as<CDenseFeatures<float64_t>>(), current);

	SG_UNREF(current);
	return ret->as<CMulticlassLabels>();
}

CRegressionLabels* CCARTree::apply_regression(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	// apply regression starting from root
	bnode_t* current=dynamic_cast<bnode_t*>(get_root());
	CLabels* ret=apply_from_current_node(dynamic_cast<CDenseFeatures<float64_t>*>(data), current);

	SG_UNREF(current);
	return ret->as<CRegressionLabels>();
}

void CCARTree::prune_using_test_dataset(CDenseFeatures<float64_t>* feats, CLabels* gnd_truth, SGVector<float64_t> weights)
{
	if (weights.vlen==0)
	{
		weights=SGVector<float64_t>(feats->get_num_vectors());
		linalg::set_const(weights, 1.0);
	}

	CDynamicObjectArray* pruned_trees=prune_tree(this);

	int32_t min_index=0;
	float64_t min_error=CMath::MAX_REAL_NUMBER;
	for (int32_t i=0;i<m_alphas->get_num_elements();++i)
	{
		CSGObject* element=pruned_trees->get_element(i);
		if (element == nullptr)
			SG_ERROR("%d element is NULL\n",i);

		bnode_t* root = dynamic_cast<bnode_t*>(element);

		CLabels* labels=apply_from_current_node(feats, root);
		float64_t error=compute_error(labels,gnd_truth,weights);
		if (error<min_error)
		{
			min_index=i;
			min_error=error;
		}

		SG_UNREF(labels);
		SG_UNREF(element);
	}

	CSGObject* element=pruned_trees->get_element(min_index);
	if (element == nullptr)
		SG_ERROR("%d element is NULL\n",min_index);

	bnode_t* root = dynamic_cast<bnode_t*>(element);
	this->set_root(root);

	SG_UNREF(pruned_trees);
	SG_UNREF(element);
}

void CCARTree::set_weights(SGVector<float64_t> w)
{
	m_weights=w;
	m_weights_set=true;
}

SGVector<float64_t> CCARTree::get_weights() const
{
	return m_weights;
}

void CCARTree::clear_weights()
{
	m_weights=SGVector<float64_t>();
	m_weights_set=false;
}

void CCARTree::set_feature_types(SGVector<bool> ft)
{
	m_nominal=ft;
	m_types_set=true;
}

SGVector<bool> CCARTree::get_feature_types() const
{
	return m_nominal;
}

void CCARTree::clear_feature_types()
{
	m_nominal=SGVector<bool>();
	m_types_set=false;
}

int32_t CCARTree::get_num_folds() const
{
	return m_folds;
}

void CCARTree::set_num_folds(int32_t folds)
{
	REQUIRE(folds>1,"Number of folds is expected to be greater than 1. Supplied value is %d\n",folds)
	m_folds=folds;
}

int32_t CCARTree::get_max_depth() const
{
	return m_max_depth;
}

void CCARTree::set_max_depth(int32_t depth)
{
	REQUIRE(depth>0,"Max allowed tree depth should be greater than 0. Supplied value is %d\n",depth)
	m_max_depth=depth;
}

int32_t CCARTree::get_min_node_size() const
{
	return m_min_node_size;
}

void CCARTree::set_min_node_size(int32_t nsize)
{
	REQUIRE(nsize>0,"Min allowed node size should be greater than 0. Supplied value is %d\n",nsize)
	m_min_node_size=nsize;
}

void CCARTree::set_label_epsilon(float64_t ep)
{
	REQUIRE(ep>=0,"Input epsilon value is expected to be greater than or equal to 0\n")
	m_label_epsilon=ep;
}

bool CCARTree::train_machine(CFeatures* data)
{
	REQUIRE(data,"Data required for training\n")
	REQUIRE(data->get_feature_class()==C_DENSE,"Dense data required for training\n")

	auto dense_features = data->as<CDenseFeatures<float64_t>>();
	auto num_features = dense_features->get_num_features();
	auto num_vectors = dense_features->get_num_vectors();

	if (m_weights_set)
	{
		REQUIRE(m_weights.vlen==num_vectors,"Length of weights vector (currently %d) should be same as"
					" number of vectors in data (presently %d)",m_weights.vlen,num_vectors)
	}
	else
	{
		// all weights are equal to 1
		m_weights=SGVector<float64_t>(num_vectors);
		linalg::set_const(m_weights, 1.0);
	}

	if (m_types_set)
	{
		REQUIRE(
		    m_nominal.vlen == num_features,
		    "Length of m_nominal vector (currently %d) should "
		    "be same as number of features in data (presently %d).\n",
		    m_nominal.vlen, num_features)
	}
	else
	{
		SG_WARNING(
		    "Feature types are not specified. All features are "
		    "considered as continuous in training.\n")
		m_nominal=SGVector<bool>(num_features);
		linalg::set_const(m_nominal, false);
	}

	auto dense_labels = m_labels->as<CDenseLabels>();
	set_root(CARTtrain(dense_features,m_weights,dense_labels,0));

	if (m_apply_cv_pruning)
	{
		prune_by_cross_validation(dense_features,m_folds);
	}

	return true;
}

void CCARTree::set_sorted_features(SGMatrix<float64_t>& sorted_feats, SGMatrix<index_t>& sorted_indices)
{
	m_pre_sort=true;
	m_sorted_features=sorted_feats;
	m_sorted_indices=sorted_indices;
}

void CCARTree::pre_sort_features(CFeatures* data, SGMatrix<float64_t>& sorted_feats, SGMatrix<index_t>& sorted_indices)
{
	SGMatrix<float64_t> mat=(data)->as<CDenseFeatures<float64_t>>()->get_feature_matrix();
	sorted_feats = SGMatrix<float64_t>(mat.num_cols, mat.num_rows);
	sorted_indices = SGMatrix<index_t>(mat.num_cols, mat.num_rows);
	for(int32_t i=0; i<sorted_indices.num_cols; i++)
		for(int32_t j=0; j<sorted_indices.num_rows; j++)
			sorted_indices(j,i)=j;

	Map<MatrixXd> map_sorted_feats(sorted_feats.matrix, mat.num_cols, mat.num_rows);
	Map<MatrixXd> map_data(mat.matrix, mat.num_rows, mat.num_cols);

	map_sorted_feats=map_data.transpose();

	#pragma omp parallel for
	for(int32_t i=0; i<sorted_feats.num_cols; i++)
		CMath::qsort_index(sorted_feats.get_column_vector(i), sorted_indices.get_column_vector(i), sorted_feats.num_rows);

}

CBinaryTreeMachineNode<CARTreeNodeData>* CCARTree::CARTtrain(CDenseFeatures<float64_t>* data, const SGVector<float64_t>& weights, CDenseLabels* labels, int32_t level)
{
	REQUIRE(labels,"labels have to be supplied\n");
	REQUIRE(data,"data matrix has to be supplied\n");

	bnode_t* node=new bnode_t();
	auto labels_vec = labels->get_labels();
	auto mat = data->get_feature_matrix();
	auto num_feats=mat.num_rows;
	auto num_vecs=mat.num_cols;

	// calculate node label
	switch(m_mode)
	{
		case PT_REGRESSION:
			{
				float64_t sum = linalg::dot(labels_vec, weights);
				// lsd*total_weight=sum_of_squared_deviation
				float64_t tot=0;
				node->data.weight_minus_node=tot*least_squares_deviation(labels_vec,weights,tot);
				node->data.node_label=sum/tot;
				node->data.total_weight=tot;

				break;
			}
		case PT_MULTICLASS:
			{
				SGVector<float64_t> lab=labels_vec.clone();
				std::sort(lab.begin(), lab.end());
				// stores max total weight for a single label
				auto max=weights[0];
				// stores one of the indices having max total weight
				index_t maxi=0;
				auto c=weights[0];
				for (index_t i=1;i<lab.vlen;++i)
				{
					if (lab[i]==lab[i-1])
					{
						c+=weights[i];
					}
					else if (c>max)
					{
						max=c;
						maxi=i-1;
						c=weights[i];
					}
					else
					{
						c=weights[i];
					}
				}

				if (c>max)
				{
					max=c;
					maxi=lab.vlen-1;
				}

				node->data.node_label=lab[maxi];

				// resubstitution error calculation
				node->data.total_weight=linalg::sum(weights);
				node->data.weight_minus_node=node->data.total_weight-max;
				break;
			}
		default :
			SG_ERROR("mode should be either PT_MULTICLASS or PT_REGRESSION\n");
	}

	// check stopping rules
	// case 1 : max tree depth reached if max_depth set
	if ((m_max_depth>0) && (level==m_max_depth))
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	// case 2 : min node size violated if min_node_size specified
	if ((m_min_node_size>1) && (labels_vec.vlen<=m_min_node_size))
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	// choose best attribute
	// transit_into_values for left child
	SGVector<float64_t> left(num_feats);
	// transit_into_values for right child
	SGVector<float64_t> right(num_feats);
	// final data distribution among children
	SGVector<bool> left_final(num_vecs);
	int32_t num_missing_final=0;
	int32_t c_left=-1;
	int32_t c_right=-1;
	int32_t best_attribute;

	SGVector<index_t> indices(num_vecs);
	if (m_pre_sort)
	{
		CSubsetStack* subset_stack = data->get_subset_stack();
		if (subset_stack->has_subsets())
			indices=(subset_stack->get_last_subset())->get_subset_idx();
		else
			linalg::range_fill(indices);
		SG_UNREF(subset_stack);
		best_attribute=compute_best_attribute(m_sorted_features,weights,labels,left,right,left_final,num_missing_final,c_left,c_right,0,indices);
	}
	else
		best_attribute=compute_best_attribute(mat,weights,labels,left,right,left_final,num_missing_final,c_left,c_right);

	if (best_attribute==-1)
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	SGVector<float64_t> left_transit(c_left);
	SGVector<float64_t> right_transit(c_right);
	sg_memcpy(left_transit.vector,left.vector,c_left*sizeof(float64_t));
	sg_memcpy(right_transit.vector,right.vector,c_right*sizeof(float64_t));

	if (num_missing_final>0)
	{
		SGVector<bool> is_left_final(num_vecs-num_missing_final);
		int32_t ilf=0;
		for (int32_t i=0;i<num_vecs;++i)
		{
			if (mat(best_attribute,i)!=MISSING)
				is_left_final[ilf++]=left_final[i];
		}

		left_final=surrogate_split(mat,weights,is_left_final,best_attribute);
	}

	index_t count_left=std::count(left_final.begin(), left_final.end(), true);

	SGVector<index_t> subsetl(count_left);
	SGVector<float64_t> weightsl(count_left);
	SGVector<index_t> subsetr(num_vecs-count_left);
	SGVector<float64_t> weightsr(num_vecs-count_left);
	index_t l=0;
	index_t r=0;
	for (index_t c = 0; c < num_vecs; ++c)
	{
		if (left_final[c])
		{
			subsetl[l]=c;
			weightsl[l++]=weights[c];
		}
		else
		{
			subsetr[r]=c;
			weightsr[r++]=weights[c];
		}
	}

	// left child
	auto feats_train = view(data, subsetl);
	auto labels_train = view(labels, subsetl);
	bnode_t* left_child =
	    CARTtrain(feats_train, weightsl, labels_train, level + 1);

	// right child
	feats_train = view(data, subsetr);
	labels_train = view(labels, subsetr);
	bnode_t* right_child =
	    CARTtrain(feats_train, weightsr, labels_train, level + 1);

	// set node parameters
	node->data.attribute_id=best_attribute;
	node->left(left_child);
	node->right(right_child);
	left_child->data.transit_into_values=left_transit;
	right_child->data.transit_into_values=right_transit;
	node->data.num_leaves=left_child->data.num_leaves+right_child->data.num_leaves;
	node->data.weight_minus_branch=left_child->data.weight_minus_branch+right_child->data.weight_minus_branch;

	return node;
}

SGVector<float64_t> CCARTree::get_unique_labels(const SGVector<float64_t>& labels_vec, index_t &n_ulabels) const
{
	float64_t delta=0;
	if (m_mode==PT_REGRESSION)
		delta=m_label_epsilon;

	SGVector<float64_t> ulabels(labels_vec.vlen);
	SGVector<index_t> sidx=CMath::argsort(labels_vec);
	ulabels[0]=labels_vec[sidx[0]];
	n_ulabels=1;
	index_t start=0;
	for (index_t i=1;i<sidx.vlen;i++)
	{
		if (labels_vec[sidx[i]]<=labels_vec[sidx[start]]+delta)
			continue;

		start=i;
		ulabels[n_ulabels]=labels_vec[sidx[i]];
		n_ulabels++;
	}

	return ulabels;
}

index_t CCARTree::compute_best_attribute(const SGMatrix<float64_t>& mat, const SGVector<float64_t>& weights, CDenseLabels* labels,
	SGVector<float64_t>& left, SGVector<float64_t>& right, SGVector<bool>& is_left_final, index_t &num_missing_final, index_t &count_left,
	index_t &count_right, index_t subset_size, const SGVector<index_t>& active_indices)
{
	auto labels_vec=labels->get_labels();
	auto num_vecs=labels->get_num_labels();
	auto num_feats = (m_pre_sort) ? mat.num_cols : mat.num_rows;

	index_t n_ulabels;
	auto ulabels = get_unique_labels(labels_vec, n_ulabels);

	// if all labels same early stop
	if (n_ulabels==1)
		return -1;

	float64_t delta=0;
	if (m_mode==PT_REGRESSION)
		delta=m_label_epsilon;

	SGVector<float64_t> total_wclasses(n_ulabels);
	linalg::zero(total_wclasses);

	SGVector<index_t> simple_labels(num_vecs);
	for (index_t i=0;i<num_vecs;i++)
	{
		for (index_t j=0;j<n_ulabels;j++)
		{
			if (std::abs(labels_vec[i]-ulabels[j])<=delta)
			{
				simple_labels[i]=j;
				total_wclasses[j]+=weights[i];
				break;
			}
		}
	}

	SGVector<index_t> idx(num_feats);
	linalg::range_fill(idx);
	if (subset_size)
	{
		num_feats=subset_size;
		CMath::permute(idx);
	}

	float64_t max_gain=MIN_SPLIT_GAIN;
	index_t best_attribute=-1;
	float64_t best_threshold=0;

	SGVector<int64_t> indices_mask;
	SGVector<index_t> count_indices(mat.num_rows);
	count_indices.zero();
	SGVector<index_t> dupes(num_vecs);
	linalg::range_fill(dupes);
	if (m_pre_sort)
	{
		indices_mask = SGVector<int64_t>(mat.num_rows);
		linalg::set_const(indices_mask, int64_t(-1));
		for(index_t j=0;j<active_indices.size();++j)
		{
			if (indices_mask[active_indices[j]]>=0)
				dupes[indices_mask[active_indices[j]]]=j;

			indices_mask[active_indices[j]]=j;
			count_indices[active_indices[j]]++;
		}
	}

	for (index_t i=0;i<num_feats;++i)
	{
		SGVector<float64_t> feats(num_vecs);
		SGVector<index_t> sorted_args(num_vecs);
		SGVector<index_t> temp_count_indices(count_indices.size());
		sg_memcpy(temp_count_indices.vector, count_indices.vector, sizeof(index_t)*count_indices.size());

		if (m_pre_sort)
		{
			SGVector<float64_t> temp_col(mat.get_column_vector(idx[i]), mat.num_rows, false);
			SGVector<index_t> sorted_indices(m_sorted_indices.get_column_vector(idx[i]), mat.num_rows, false);
			index_t count=0;
			for(index_t j=0;j<mat.num_rows;j++)
			{
				if (indices_mask[sorted_indices[j]]>=0)
				{
					index_t count_index = count_indices[sorted_indices[j]];
					while(count_index>0)
					{
						feats[count]=temp_col[j];
						sorted_args[count]=indices_mask[sorted_indices[j]];
						++count;
						--count_index;
					}
					if (count==num_vecs)
						break;
				}
			}
		}
		else
		{
			for (index_t j=0;j<feats.vlen;++j)
				feats[j]=mat(idx[i],j);

			// O(N*logN)
			linalg::range_fill(sorted_args);
			CMath::qsort_index(feats.vector, sorted_args.vector, feats.size());
		}
		auto n_nm_vecs = feats.vlen;
		// number of non-missing vecs
		while (feats[n_nm_vecs-1] == MISSING)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]-=weights[sorted_args[n_nm_vecs-1]];
			--n_nm_vecs;
		}

		// if only one unique value - it cannot be used to split
		if (feats[n_nm_vecs-1]<=feats[0]+EQ_DELTA)
			continue;

		if (m_nominal[idx[i]])
		{
			SGVector<index_t> simple_feats(num_vecs);
			linalg::set_const(simple_feats, -1);

			// convert to simple values
			simple_feats[0]=0;
			index_t c=0;
			for (index_t j=1;j<n_nm_vecs;++j)
			{
				if (feats[j]==feats[j-1])
					simple_feats[j]=c;
				else
					simple_feats[j]=(++c);
			}

			// collect the unique categorical values
			SGVector<float64_t> ufeats(c+1);
			ufeats[0]=feats[0];
			index_t u=0;
			for (index_t j = 1; j < n_nm_vecs; ++j)
			{
				if (feats[j]==feats[j-1])
					continue;
				else
					ufeats[++u]=feats[j];
			}

			// FIXME: this approach is way too vanilla!
			// test all 2^(I-1)-1 possible division between two nodes
			index_t num_cases=CMath::pow(2,c);
			for (index_t k = 1; k < num_cases; ++k)
			{
				SGVector<float64_t> wleft(n_ulabels), wright(n_ulabels);
				linalg::zero(wleft);
				linalg::zero(wright);

				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs);
				linalg::set_const(is_left, false);

				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(c+1);

				// fill feats_left in a unique way corresponding to the case
				for (index_t p = 0; p < feats_left.vlen; ++p)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (index_t j = 0; j < n_nm_vecs; ++j)
				{
					is_left[sorted_args[j]]=feats_left[simple_feats[j]];
					if (is_left[sorted_args[j]])
						wleft[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
					else
						wright[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
				}
				for (index_t j = n_nm_vecs-1 ; j >= 0; --j)
				{
					if(dupes[j]!=j)
						is_left[j]=is_left[dupes[j]];
				}

				float64_t g=0;
				switch(m_mode)
				{
					case PT_MULTICLASS:
						g=gain(wleft,wright,total_wclasses);
						break;
					case PT_REGRESSION:
						g=gain(wleft,wright,total_wclasses,ulabels);
						break;
					default:
						SG_ERROR("Undefined problem statement\n");
				}

				if (g>max_gain)
				{
					best_attribute=idx[i];
					max_gain=g;
					sg_memcpy(is_left_final.vector,is_left.vector,is_left.vlen*sizeof(bool));
					num_missing_final=num_vecs-n_nm_vecs;

					count_left = std::count(feats_left.begin(), feats_left.end(), true);
					count_right = c+1-count_left;

					index_t l=0;
					index_t r=0;
					if (right.vlen < count_right)
						right.resize_vector(count_right);
					for (index_t w = 0; w < feats_left.vlen; ++w)
					{
						if (feats_left[w])
							left[l++]=ufeats[w];
						else
							right[r++]=ufeats[w];
					}
				}
			}
		}
		else
		{
			// O(N)
			SGVector<float64_t> right_wclasses=total_wclasses.clone();
			SGVector<float64_t> left_wclasses(n_ulabels);
			linalg::zero(left_wclasses);

			// O(N)
			// find best split for non-nominal attribute - choose threshold (z)
			float64_t z=feats[0];
			right_wclasses[simple_labels[sorted_args[0]]]-=weights[sorted_args[0]];
			left_wclasses[simple_labels[sorted_args[0]]]+=weights[sorted_args[0]];
			for (index_t j=1;j<n_nm_vecs;++j)
			{
				if (feats[j]<=z+EQ_DELTA)
				{
					right_wclasses[simple_labels[sorted_args[j]]]-=weights[sorted_args[j]];
					left_wclasses[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
					continue;
				}
				// O(F)
				float64_t g=0;
				if (m_mode==PT_MULTICLASS)
					g=gain(left_wclasses,right_wclasses,total_wclasses);
				else if (m_mode==PT_REGRESSION)
					g=gain(left_wclasses,right_wclasses,total_wclasses,ulabels);
				else
					SG_ERROR("Undefined problem statement\n");

				if (g>max_gain)
				{
					max_gain=g;
					best_attribute=idx[i];
					best_threshold=z;
					num_missing_final=num_vecs-n_nm_vecs;
				}

				z=feats[j];
				if (feats[n_nm_vecs-1]<=z+EQ_DELTA)
					break;
				right_wclasses[simple_labels[sorted_args[j]]]-=weights[sorted_args[j]];
				left_wclasses[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
			}
		}

		// restore total_wclasses
		while (n_nm_vecs<feats.vlen)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]+=weights[sorted_args[n_nm_vecs-1]];
			++n_nm_vecs;
		}
	}

	if (best_attribute==-1)
		return -1;

	if (!m_nominal[best_attribute])
	{
		left[0]=best_threshold;
		right[0]=best_threshold;
		count_left=1;
		count_right=1;
		if (m_pre_sort)
		{
			SGVector<float64_t> temp_vec(mat.get_column_vector(best_attribute), mat.num_rows, false);
			SGVector<index_t> sorted_indices(m_sorted_indices.get_column_vector(best_attribute), mat.num_rows, false);
			index_t count=0;
			for(index_t i=0;i<mat.num_rows;++i)
			{
				if (indices_mask[sorted_indices[i]]>=0)
				{
					is_left_final[indices_mask[sorted_indices[i]]]=(temp_vec[i]<=best_threshold);
					++count;
					if (count==num_vecs)
						break;
				}
			}
			for (index_t i=num_vecs-1;i>=0;--i)
			{
				if(dupes[i]!=i)
					is_left_final[i]=is_left_final[dupes[i]];
			}

		}
		else
		{
			for (index_t i=0;i<num_vecs;++i)
				is_left_final[i]=(mat(best_attribute,i)<=best_threshold);
		}
	}

	return best_attribute;
}

SGVector<bool> CCARTree::surrogate_split(SGMatrix<float64_t> m,SGVector<float64_t> weights, SGVector<bool> nm_left, int32_t attr) const
{
	// return vector - left/right belongingness
	SGVector<bool> ret(m.num_cols);

	// ditribute data with known attributes
	index_t l=0;
	float64_t p_l=0.;
	float64_t total=0.;
	// stores indices of vectors with missing attribute
	std::vector<index_t> missing_vecs;
	// stores lambda values corresponding to missing vectors - initialized all with 0
	std::vector<float64_t> association_index;
	for (index_t i=0;i<m.num_cols;++i)
	{
		if (!CMath::fequals(m(attr,i),MISSING,0))
		{
			ret[i]=nm_left[l];
			total+=weights[i];
			if (nm_left[l++])
				p_l+=weights[i];
		}
		else
		{
			missing_vecs.push_back(i);
			association_index.push_back(0.);
		}
	}

	// for lambda calculation
	float64_t p_r=(total-p_l)/total;
	p_l/=total;
	float64_t p=std::min(p_r,p_l);

	// for each attribute (X') alternative to best split (X)
	for (index_t i = 0; i < m.num_rows; ++i)
	{
		if (i==attr)
			continue;

		// find set of vectors with non-missing values for both X and X'
		std::vector<index_t> intersect_vecs;
		for (index_t j=0;j<m.num_cols;j++)
		{
			if (!(CMath::fequals(m(i,j),MISSING,0) || CMath::fequals(m(attr,j),MISSING,0)))
				intersect_vecs.push_back(j);
		}

		if (intersect_vecs.size() == 0)
		{
			intersect_vecs.clear();
			continue;
		}


		if (m_nominal[i])
			handle_missing_vecs_for_nominal_surrogate(m,missing_vecs,association_index,intersect_vecs,ret,weights,p,i);
		else
			handle_missing_vecs_for_continuous_surrogate(m,missing_vecs,association_index,intersect_vecs,ret,weights,p,i);
	}

	// if some missing attribute vectors are yet not addressed, use majority rule
	for (size_t i=0;i<association_index.size();++i)
	{
		if (association_index.at(i)==0.)
			ret[missing_vecs.at(i)]=(p_l>=p_r);
	}

	return ret;
}

void CCARTree::handle_missing_vecs_for_continuous_surrogate(SGMatrix<float64_t> m, const std::vector<index_t>& missing_vecs,
		std::vector<float64_t>& association_index, std::vector<index_t>& intersect_vecs,
		SGVector<bool> is_left, SGVector<float64_t> weights, float64_t p, index_t attr) const
{
	// for lambda calculation - total weight of all vectors in X intersect X'
	float64_t denom=0.;
	SGVector<float64_t> feats(intersect_vecs.size());
	for (index_t j = 0; j < intersect_vecs.size(); ++j)
	{
		feats[j]=m(attr,intersect_vecs.at(j));
		denom+=weights[intersect_vecs.at(j)];
	}

	// unique feature values for X'
	index_t num_unique=feats.unique(feats.vector,feats.vlen);

	// all possible splits for chosen attribute
	for (index_t j = 0; j < num_unique-1; ++j)
	{
		auto z=feats[j];
		float64_t numer=0.;
		float64_t numerc=0.;
		for (size_t k = 0; k < intersect_vecs.size(); ++k)
		{
			// if both go left or both go right
			if ((m(attr,intersect_vecs.at(k))<=z) && is_left[intersect_vecs.at(k)])
				numer+=weights[intersect_vecs.at(k)];
			else if ((m(attr,intersect_vecs.at(k))>z) && !is_left[intersect_vecs.at(k)])
				numer+=weights[intersect_vecs.at(k)];
			// complementary split cases - one goes left other right
			else if ((m(attr,intersect_vecs.at(k))<=z) && !is_left[intersect_vecs.at(k)])
				numerc+=weights[intersect_vecs.at(k)];
			else if ((m(attr,intersect_vecs.at(k))>z) && is_left[intersect_vecs.at(k)])
				numerc+=weights[intersect_vecs.at(k)];
		}

		float64_t lambda = (numer>=numerc)
			? (p-(1-numer/denom))/p
			: (p-(1-numerc/denom))/p;

		for (size_t k = 0; k < missing_vecs.size(); ++k)
		{
			if ((lambda>association_index.at(k)) &&
			(!CMath::fequals(m(attr,missing_vecs.at(k)),MISSING,0)))
			{
				association_index[k] = lambda;
				is_left[missing_vecs.at(k)] = (numer>=numerc)
					? (m(attr,missing_vecs.at(k))<=z)
					: (m(attr,missing_vecs.at(k))>z);
			}
		}
	}
}

void CCARTree::handle_missing_vecs_for_nominal_surrogate(SGMatrix<float64_t> m, const std::vector<index_t>& missing_vecs,
		std::vector<float64_t>& association_index, const std::vector<index_t>& intersect_vecs,
		SGVector<bool> is_left, SGVector<float64_t> weights, float64_t p, index_t attr) const
{
	// for lambda calculation - total weight of all vectors in X intersect X'
	float64_t denom=0.;
	SGVector<float64_t> feats(intersect_vecs.size());
	for (index_t j = 0; j < intersect_vecs.size(); ++j)
	{
		feats[j]=m(attr,intersect_vecs.at(j));
		denom+=weights[intersect_vecs.at(j)];
	}

	// unique feature values for X'
	index_t num_unique = feats.unique(feats.vector,feats.vlen);

	// scan all splits for chosen alternative attribute X'
	auto num_cases=CMath::pow(2,(num_unique-1));
	for (int32_t j = 1; j < num_cases; ++j)
	{
		SGVector<bool> feats_left(num_unique);
		for (int32_t k = 0; k < num_unique; ++k)
			feats_left[k]=((j/CMath::pow(2,k))%(CMath::pow(2,k+1))==1);

		SGVector<bool> intersect_vecs_left(intersect_vecs.size());
		for (int32_t k=0;k<intersect_vecs.size();++k)
		{
			for (int32_t q=0;q<num_unique;++q)
			{
				if (feats[q]==m(attr,intersect_vecs.at(k)))
				{
					intersect_vecs_left[k]=feats_left[q];
					break;
				}
			}
		}

		float64_t numer=0.;
		float64_t numerc=0.;
		for (int32_t k=0;k<intersect_vecs.size();++k)
		{
			// if both go left or both go right
			if (intersect_vecs_left[k]==is_left[intersect_vecs.at(k)])
				numer+=weights[intersect_vecs.at(k)];
			else
				numerc+=weights[intersect_vecs.at(k)];
		}

		// lambda for this split (2 case identical split/complementary split)
		float64_t lambda = (numer>=numerc)
			? (p-(1-numer/denom))/p
			: (p-(1-numerc/denom))/p;

		// address missing value vectors not yet addressed or addressed using worse split
		for (size_t k = 0; k < missing_vecs.size(); ++k)
		{
			if ((lambda>association_index.at(k)) &&
			(!CMath::fequals(m(attr,missing_vecs.at(k)),MISSING,0)))
			{
				association_index[k] = lambda;
				// decide left/right based on which feature value the chosen data point has
				for (index_t q = 0; q < num_unique; ++q)
				{
					if (feats[q]==m(attr,missing_vecs.at(k)))
					{
						is_left[missing_vecs.at(k)] = (numer>=numerc)
							? feats_left[q]
							: !feats_left[q];

						break;
					}
				}
			}
		}
	}
}

float64_t CCARTree::gain(const SGVector<float64_t>& wleft, const SGVector<float64_t>& wright, const SGVector<float64_t>& wtotal,
						const SGVector<float64_t>& feats) const
{
	float64_t total_lweight=0;
	float64_t total_rweight=0;
	float64_t total_weight=0;

	float64_t lsd_n=least_squares_deviation(feats,wtotal,total_weight);
	float64_t lsd_l=least_squares_deviation(feats,wleft,total_lweight);
	float64_t lsd_r=least_squares_deviation(feats,wright,total_rweight);

	return lsd_n-(lsd_l*(total_lweight/total_weight))-(lsd_r*(total_rweight/total_weight));
}

float64_t CCARTree::gain(const SGVector<float64_t>& wleft, const SGVector<float64_t>& wright, const SGVector<float64_t>& wtotal) const
{
	float64_t total_lweight=0;
	float64_t total_rweight=0;
	float64_t total_weight=0;

	float64_t gini_n=gini_impurity_index(wtotal,total_weight);
	float64_t gini_l=gini_impurity_index(wleft,total_lweight);
	float64_t gini_r=gini_impurity_index(wright,total_rweight);
	return gini_n-(gini_l*(total_lweight/total_weight))-(gini_r*(total_rweight/total_weight));
}

float64_t CCARTree::gini_impurity_index(const SGVector<float64_t>& weighted_lab_classes, float64_t &total_weight) const
{
	total_weight = linalg::sum(weighted_lab_classes);
	float64_t gini = linalg::dot(weighted_lab_classes, weighted_lab_classes);
	return 1.0-(gini/(total_weight*total_weight));
}

float64_t CCARTree::least_squares_deviation(const SGVector<float64_t>& feats, const SGVector<float64_t>& weights, float64_t &total_weight) const
{
	SGVector<float64_t> wrap_feats(feats.vector, weights.size(), false);
	float64_t mean = linalg::dot(weights, wrap_feats);
	total_weight = linalg::sum(weights);

	mean/=total_weight;
	float64_t dev=0;
	for (index_t i=0;i<weights.vlen;++i)
		dev+=weights[i]*(feats[i]-mean)*(feats[i]-mean);

	return dev/total_weight;
}

CLabels* CCARTree::apply_from_current_node(CDenseFeatures<float64_t>* feats, bnode_t* current)
{
	auto num_vecs=feats->get_num_vectors();
	REQUIRE(num_vecs>0, "No data provided in apply\n");

	SGVector<float64_t> labels(num_vecs);
	for (index_t i=0;i<num_vecs;++i)
	{
		auto sample=feats->get_feature_vector(i);
		bnode_t* node=current;
		SG_REF(node);

		// until leaf is reached
		while(node->data.num_leaves!=1)
		{
			bnode_t* leftchild=node->left();

			if (m_nominal[node->data.attribute_id])
			{
				SGVector<float64_t> comp=leftchild->data.transit_into_values;
				bool flag=false;
				for (index_t k=0;k<comp.vlen;++k)
				{
					if (comp[k]==sample[node->data.attribute_id])
					{
						flag=true;
						break;
					}
				}

				if (flag)
				{
					SG_UNREF(node);
					node=leftchild;
					SG_REF(leftchild);
				}
				else
				{
					SG_UNREF(node);
					node=node->right();
				}
			}
			else
			{
				if (sample[node->data.attribute_id]<=leftchild->data.transit_into_values[0])
				{
					SG_UNREF(node);
					node=leftchild;
					SG_REF(leftchild);
				}
				else
				{
					SG_UNREF(node);
					node=node->right();
				}
			}

			SG_UNREF(leftchild);
		}

		labels[i]=node->data.node_label;
		SG_UNREF(node);
	}

	switch(m_mode)
	{
		case PT_MULTICLASS:
		{
			CMulticlassLabels* mlabels=new CMulticlassLabels(labels);
			return mlabels;
		}

		case PT_REGRESSION:
		{
			CRegressionLabels* rlabels=new CRegressionLabels(labels);
			return rlabels;
		}

		default:
			SG_ERROR("mode should be either PT_MULTICLASS or PT_REGRESSION\n");
	}

	return NULL;
}

void CCARTree::prune_by_cross_validation(CDenseFeatures<float64_t>* data, int32_t folds)
{
	auto num_vecs=data->get_num_vectors();

	// divide data into V folds randomly
	SGVector<index_t> subid(num_vecs);
	subid.random_vector(subid.vector,subid.vlen,0,folds-1);

	// for each fold subset
	std::vector<float64_t> r_cv;
	std::vector<float64_t> alphak;
	SGVector<int32_t> num_alphak(folds);
	for (int32_t i=0;i<folds;++i)
	{
		// for chosen fold, create subset for training parameters
		std::vector<index_t> test_indices;
		std::vector<index_t> train_indices;
		for (index_t j = 0; j < num_vecs; ++j)
		{
			(subid[j]==i)
				? test_indices.push_back(j)
				: train_indices.push_back(j);
		}

		if (test_indices.size()==0 || train_indices.size()==0)
		{
			SG_ERROR("Unfortunately you have reached the very low probability event where atleast one of "
					"the subsets in cross-validation is not represented at all. Please re-run.")
		}

		SGVector<int32_t> subset(train_indices.data(),train_indices.size(),false);
		auto dense_labels = m_labels->as<CDenseLabels>();
		auto feats_train = view(data, subset);
		auto labels_train = view(dense_labels, subset);
		SGVector<float64_t> subset_weights(train_indices.size());

		for (index_t j = 0; j < train_indices.size(); ++j)
			subset_weights[j]=m_weights[train_indices.at(j)];

		// train with training subset
		bnode_t* root = CARTtrain(feats_train, subset_weights, labels_train, 0);

		// prune trained tree
		CTreeMachine<CARTreeNodeData>* tmax=new CTreeMachine<CARTreeNodeData>();
		tmax->set_root(root);
		CDynamicObjectArray* pruned_trees=prune_tree(tmax);

		subset=SGVector<int32_t>(test_indices.data(),test_indices.size(),false);
		feats_train = view(data, subset);
		labels_train = view(dense_labels, subset);
		subset_weights=SGVector<float64_t>(test_indices.size());
		for (int32_t j=0;j<test_indices.size();++j)
			subset_weights[j]=m_weights[test_indices.at(j)];

		// calculate R_CV values for each alpha_k using test subset and store them
		num_alphak[i]=m_alphas->get_num_elements();
		for (int32_t j=0;j<m_alphas->get_num_elements();++j)
		{
			alphak.push_back(m_alphas->get_element(j));
			CSGObject* jth_element=pruned_trees->get_element(j);
			bnode_t* current_root=NULL;
			if (jth_element!=NULL)
				current_root=dynamic_cast<bnode_t*>(jth_element);
			else
				SG_ERROR("%d element is NULL which should not be",j);

			CLabels* labels =
			    apply_from_current_node(feats_train, current_root);
			float64_t error =
			    compute_error(labels, labels_train, subset_weights);
			r_cv.push_back(error);
			SG_UNREF(labels);
			SG_UNREF(jth_element);
		}

		SG_UNREF(tmax);
		SG_UNREF(pruned_trees);
	}

	// prune the original T_max
	CDynamicObjectArray* pruned_trees=prune_tree(this);

	// find subtree with minimum R_cv
	int32_t min_index=-1;
	float64_t min_r_cv=CMath::MAX_REAL_NUMBER;
	for (int32_t i=0;i<m_alphas->get_num_elements();++i)
	{
		float64_t alpha=0.;
		if (i==m_alphas->get_num_elements()-1)
			alpha=m_alphas->get_element(i)+1;
		else
			alpha = std::sqrt(
			    m_alphas->get_element(i) * m_alphas->get_element(i + 1));

		float64_t rv=0.;
		int32_t base=0;
		for (int32_t j=0;j<folds;++j)
		{
			bool flag=false;
			for (int32_t k=base;k<num_alphak[j]+base-1;++k)
			{
				if (alphak.at(k)<=alpha && alphak.at(k+1)>alpha)
				{
					rv+=r_cv.at(k);
					flag=true;
					break;
				}
			}

			if (!flag)
				rv+=r_cv.at(num_alphak[j]+base-1);

			base+=num_alphak[j];
		}

		if (rv<min_r_cv)
		{
			min_index=i;
			min_r_cv=rv;
		}
	}

	CSGObject* element=pruned_trees->get_element(min_index);
	bnode_t* best_tree_root=NULL;
	if (element==nullptr)
		SG_ERROR("%d element is NULL which should not be",min_index);

	best_tree_root=dynamic_cast<bnode_t*>(element);
	this->set_root(best_tree_root);

	SG_UNREF(element);
	SG_UNREF(pruned_trees);
}

float64_t CCARTree::compute_error(CLabels* labels, CLabels* reference, SGVector<float64_t> weights) const
{
	REQUIRE(labels,"input labels cannot be NULL");
	REQUIRE(reference,"reference labels cannot be NULL")

	CDenseLabels* gnd_truth = reference->as<CDenseLabels>();
	CDenseLabels* result = labels->as<CDenseLabels>();

	float64_t denom=linalg::sum(weights);
	float64_t numer=0.;
	switch (m_mode)
	{
		case PT_MULTICLASS:
		{
			for (int32_t i=0;i<weights.vlen;++i)
			{
				if (gnd_truth->get_label(i)!=result->get_label(i))
					numer+=weights[i];
			}

			return numer/denom;
		}

		case PT_REGRESSION:
		{
			for (int32_t i=0;i<weights.vlen;++i)
				numer+=weights[i]*CMath::pow((gnd_truth->get_label(i)-result->get_label(i)),2);

			return numer/denom;
		}

		default:
			SG_ERROR("Case not possible\n");
	}

	return 0.;
}

CDynamicObjectArray* CCARTree::prune_tree(CTreeMachine<CARTreeNodeData>* tree)
{
	REQUIRE(tree, "Tree not provided for pruning.\n");

	CDynamicObjectArray* trees=new CDynamicObjectArray();
	SG_UNREF(m_alphas);
	m_alphas=new CDynamicArray<float64_t>();
	SG_REF(m_alphas);

	// base tree alpha_k=0
	m_alphas->push_back(0);
	CTreeMachine<CARTreeNodeData>* t1=tree->clone_tree();
	SG_REF(t1);
	node_t* t1root=t1->get_root();
	bnode_t* t1_root=NULL;
	if (t1root!=NULL)
		t1_root=dynamic_cast<bnode_t*>(t1root);
	else
		SG_ERROR("t1_root is NULL. This is not expected\n")

	form_t1(t1_root);
	trees->push_back(t1_root);
	while(t1_root->data.num_leaves>1)
	{
		CTreeMachine<CARTreeNodeData>* t2=t1->clone_tree();
		SG_REF(t2);

		node_t* t2root=t2->get_root();
		bnode_t* t2_root=NULL;
		if (t2root!=NULL)
			t2_root=dynamic_cast<bnode_t*>(t2root);
		else
			SG_ERROR("t1_root is NULL. This is not expected\n")

		float64_t a_k=find_weakest_alpha(t2_root);
		m_alphas->push_back(a_k);
		cut_weakest_link(t2_root,a_k);
		trees->push_back(t2_root);

		SG_UNREF(t1);
		SG_UNREF(t1_root);
		t1=t2;
		t1_root=t2_root;
	}

	SG_UNREF(t1);
	SG_UNREF(t1_root);
	return trees;
}

float64_t CCARTree::find_weakest_alpha(bnode_t* node) const
{
	if (node->data.num_leaves!=1)
	{
		bnode_t* left=node->left();
		bnode_t* right=node->right();

		SGVector<float64_t> weak_links(3);
		weak_links[0]=find_weakest_alpha(left);
		weak_links[1]=find_weakest_alpha(right);
		weak_links[2]=(node->data.weight_minus_node-node->data.weight_minus_branch)/node->data.total_weight;
		weak_links[2]/=(node->data.num_leaves-1.0);

		SG_UNREF(left);
		SG_UNREF(right);
		return *std::min_element(weak_links.begin(), weak_links.end());
	}

	return CMath::MAX_REAL_NUMBER;
}

void CCARTree::cut_weakest_link(bnode_t* node, float64_t alpha)
{
	if (node->data.num_leaves==1)
		return;

	float64_t g=(node->data.weight_minus_node-node->data.weight_minus_branch)/node->data.total_weight;
	g/=(node->data.num_leaves-1.0);
	if (alpha==g)
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		CDynamicObjectArray* children=new CDynamicObjectArray();
		node->set_children(children);

		SG_UNREF(children);
	}
	else
	{
		bnode_t* left=node->left();
		bnode_t* right=node->right();
		cut_weakest_link(left,alpha);
		cut_weakest_link(right,alpha);
		node->data.num_leaves=left->data.num_leaves+right->data.num_leaves;
		node->data.weight_minus_branch=left->data.weight_minus_branch+right->data.weight_minus_branch;

		SG_UNREF(left);
		SG_UNREF(right);
	}
}

void CCARTree::form_t1(bnode_t* node)
{
	if (node->data.num_leaves!=1)
	{
		bnode_t* left=node->left();
		bnode_t* right=node->right();

		form_t1(left);
		form_t1(right);

		node->data.num_leaves=left->data.num_leaves+right->data.num_leaves;
		node->data.weight_minus_branch=left->data.weight_minus_branch+right->data.weight_minus_branch;
		if (node->data.weight_minus_node==node->data.weight_minus_branch)
		{
			node->data.num_leaves=1;
			CDynamicObjectArray* children=new CDynamicObjectArray();
			node->set_children(children);

			SG_UNREF(children);
		}

		SG_UNREF(left);
		SG_UNREF(right);
	}
}

void CCARTree::init()
{
	m_nominal=SGVector<bool>();
	m_weights=SGVector<float64_t>();
	m_mode=PT_MULTICLASS;
	m_pre_sort=false;
	m_types_set=false;
	m_weights_set=false;
	m_apply_cv_pruning=false;
	m_folds=5;
	m_alphas=new CDynamicArray<float64_t>();
	SG_REF(m_alphas);
	m_max_depth=0;
	m_min_node_size=0;
	m_label_epsilon=1e-7;
	m_sorted_features=SGMatrix<float64_t>();
	m_sorted_indices=SGMatrix<index_t>();

	SG_ADD(&m_pre_sort, "pre_sort", "presort", ParameterProperties());
	SG_ADD(&m_sorted_features, "sorted_features", "sorted feats", ParameterProperties());
	SG_ADD(&m_sorted_indices, "sorted_indices", "sorted indices", ParameterProperties());
	SG_ADD(&m_nominal, "nominal", "feature types", ParameterProperties());
	SG_ADD(&m_weights, "weights", "weights", ParameterProperties());
	SG_ADD(&m_weights_set, "weights_set", "weights set", ParameterProperties());
	SG_ADD(&m_types_set, "types_set", "feature types set", ParameterProperties());
	SG_ADD(&m_apply_cv_pruning, "apply_cv_pruning", "apply cross validation pruning", ParameterProperties());
	SG_ADD(&m_folds, "folds", "number of subsets for cross validation", ParameterProperties());
	SG_ADD(&m_max_depth, "max_depth", "max allowed tree depth", ParameterProperties());
	SG_ADD(&m_min_node_size, "min_node_size", "min allowed node size", ParameterProperties());
	SG_ADD(&m_label_epsilon, "label_epsilon", "epsilon for labels", ParameterProperties());
	SG_ADD((machine_int_t*)&m_mode, "mode", "problem type (multiclass or regression)", ParameterProperties());
}
