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

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/CARTree.h>

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
		set_cv_pruning();
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
	CLabels* ret=apply_from_current_node(dynamic_cast<CDenseFeatures<float64_t>*>(data), current);

	SG_UNREF(current);
	return dynamic_cast<CMulticlassLabels*>(ret); 
}

CRegressionLabels* CCARTree::apply_regression(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	// apply regression starting from root
	bnode_t* current=dynamic_cast<bnode_t*>(get_root());
	CLabels* ret=apply_from_current_node(dynamic_cast<CDenseFeatures<float64_t>*>(data), current);

	SG_UNREF(current);
	return dynamic_cast<CRegressionLabels*>(ret);
}

void CCARTree::prune_using_test_dataset(CDenseFeatures<float64_t>* feats, CLabels* gnd_truth, SGVector<float64_t> weights)
{
	if (weights.vlen==0)
	{
		weights=SGVector<float64_t>(feats->get_num_vectors());
		weights.fill_vector(weights.vector,weights.vlen,1);
	}

	CDynamicObjectArray* pruned_trees=prune_tree(this);

	int32_t min_index=0;
	float64_t min_error=CMath::MAX_REAL_NUMBER;
	for (int32_t i=0;i<m_alphas->get_num_elements();i++)
	{
		CSGObject* element=pruned_trees->get_element(i);
		bnode_t* root=NULL;
		if (element!=NULL)
			root=dynamic_cast<bnode_t*>(element);
		else
			SG_ERROR("%d element is NULL\n",i);

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
	bnode_t* root=NULL;
	if (element!=NULL)
		root=dynamic_cast<bnode_t*>(element);
	else
		SG_ERROR("%d element is NULL\n",min_index);

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

	int32_t num_features=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_features();
	int32_t num_vectors=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_vectors();

	if (m_weights_set)
	{
		REQUIRE(m_weights.vlen==num_vectors,"Length of weights vector (currently %d) should be same as" 
					" number of vectors in data (presently %d)",m_weights.vlen,num_vectors)
	}
	else
	{
		// all weights are equal to 1
		m_weights=SGVector<float64_t>(num_vectors);
		m_weights.fill_vector(m_weights.vector,m_weights.vlen,1.0);
	}

	if (m_types_set)
	{
		REQUIRE(m_nominal.vlen==num_features,"Length of m_nominal vector (currently %d) should "
			"be same as number of features in data (presently %d)",m_nominal.vlen,num_features)
	}
	else
	{
		SG_WARNING("Feature types are not specified. All features are considered as continuous in training")
		m_nominal=SGVector<bool>(num_features);
		m_nominal.fill_vector(m_nominal.vector,m_nominal.vlen,false);
	}

	set_root(CARTtrain(data,m_weights,m_labels,0));

	if (m_apply_cv_pruning)
	{
		CDenseFeatures<float64_t>* feats=dynamic_cast<CDenseFeatures<float64_t>*>(data);
		prune_by_cross_validation(feats,m_folds);
	}

	return true;
}

CBinaryTreeMachineNode<CARTreeNodeData>* CCARTree::CARTtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels, int32_t level)
{
	REQUIRE(labels,"labels have to be supplied\n");
	REQUIRE(data,"data matrix has to be supplied\n");

	bnode_t* node=new bnode_t();
	SGVector<float64_t> labels_vec=(dynamic_cast<CDenseLabels*>(labels))->get_labels();
	SGMatrix<float64_t> mat=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_feature_matrix();
	int32_t num_feats=mat.num_rows;
	int32_t num_vecs=mat.num_cols;

	// calculate node label
	switch(m_mode)
	{
		case PT_REGRESSION:
			{
				float64_t sum=0;
				for (int32_t i=0;i<labels_vec.vlen;i++)
					sum+=labels_vec[i]*weights[i];

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
				lab.qsort();
				// stores max total weight for a single label 
				int32_t max=weights[0];
				// stores one of the indices having max total weight
				int32_t maxi=0;
				int32_t c=weights[0];
				for (int32_t i=1;i<lab.vlen;i++)
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
				node->data.total_weight=weights.sum(weights);
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

	int32_t best_attribute=compute_best_attribute(mat,weights,labels_vec,left,right,left_final,num_missing_final,c_left,c_right);

	if (best_attribute==-1)
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	SGVector<float64_t> left_transit(c_left);
	SGVector<float64_t> right_transit(c_right);
	memcpy(left_transit.vector,left.vector,c_left*sizeof(float64_t));
	memcpy(right_transit.vector,right.vector,c_right*sizeof(float64_t));

	if (num_missing_final>0)
	{
		SGVector<bool> is_left_final(num_vecs-num_missing_final);
		int32_t ilf=0;
		for (int32_t i=0;i<num_vecs;i++)
		{
			if (mat(best_attribute,i)!=MISSING)
				is_left_final[ilf++]=left_final[i];
		}

		left_final=surrogate_split(mat,weights,is_left_final,best_attribute);
	}

	int32_t count_left=0;
	for (int32_t c=0;c<num_vecs;c++)
		count_left=(left_final[c])?count_left+1:count_left;

	SGVector<index_t> subsetl(count_left);
	SGVector<float64_t> weightsl(count_left);
	SGVector<index_t> subsetr(num_vecs-count_left);
	SGVector<float64_t> weightsr(num_vecs-count_left);
	index_t l=0;
	index_t r=0;
	for (int32_t c=0;c<num_vecs;c++)
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
	data->add_subset(subsetl);
	labels->add_subset(subsetl);
	bnode_t* left_child=CARTtrain(data,weightsl,labels,level+1);
	data->remove_subset();
	labels->remove_subset();

	// right child
	data->add_subset(subsetr);
	labels->add_subset(subsetr);
	bnode_t* right_child=CARTtrain(data,weightsr,labels,level+1);
	data->remove_subset();
	labels->remove_subset();

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

SGVector<float64_t> CCARTree::get_unique_labels(SGVector<float64_t> labels_vec, int32_t &n_ulabels)
{
	float64_t delta=0;
	if (m_mode==PT_REGRESSION)
		delta=m_label_epsilon;

	SGVector<float64_t> ulabels(labels_vec.vlen);
	SGVector<index_t> sidx=labels_vec.argsort();
	ulabels[0]=labels_vec[sidx[0]];
	n_ulabels=1;
	int32_t start=0;
	for (int32_t i=1;i<sidx.vlen;i++)
	{
		if (labels_vec[sidx[i]]<=labels_vec[sidx[start]]+delta)
			continue;

		start=i;
		ulabels[n_ulabels]=labels_vec[sidx[i]];
		n_ulabels++;
	}

	return ulabels;
}

int32_t CCARTree::compute_best_attribute(SGMatrix<float64_t> mat, SGVector<float64_t> weights, SGVector<float64_t> labels_vec, 
	SGVector<float64_t> left, SGVector<float64_t> right, SGVector<bool> is_left_final, int32_t &num_missing_final, int32_t &count_left, 
	int32_t &count_right)
{
	int32_t num_vecs=mat.num_cols;
	int32_t num_feats=mat.num_rows;

	int32_t n_ulabels;
	SGVector<float64_t> ulabels=get_unique_labels(labels_vec,n_ulabels);

	// if all labels same early stop
	if (n_ulabels==1)
		return -1;

	float64_t delta=0;
	if (m_mode==PT_REGRESSION)
		delta=m_label_epsilon;

	SGVector<float64_t> total_wclasses(n_ulabels);
	total_wclasses.zero();

	SGVector<int32_t> simple_labels(num_vecs);
	for (int32_t i=0;i<num_vecs;i++)
	{
		for (int32_t j=0;j<n_ulabels;j++)
		{
			if (CMath::abs(labels_vec[i]-ulabels[j])<=delta)
			{
				simple_labels[i]=j;
				total_wclasses[j]+=weights[i];
				break;
			}
		}
	}

	float64_t max_gain=MIN_SPLIT_GAIN;
	int32_t best_attribute=-1;
	float64_t best_threshold=0;
	for (int32_t i=0;i<num_feats;i++)
	{
		SGVector<float64_t> feats(num_vecs);
		for (int32_t j=0;j<num_vecs;j++)
			feats[j]=mat(i,j);

		// O(N*logN)
		SGVector<index_t> sorted_args=feats.argsort();

		// number of non-missing vecs
		int32_t n_nm_vecs=feats.vlen;
		while (feats[sorted_args[n_nm_vecs-1]]==MISSING)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]-=weights[sorted_args[n_nm_vecs-1]];
			n_nm_vecs--;
		}

		// if only one unique value - it cannot be used to split
		if (feats[sorted_args[n_nm_vecs-1]]<=feats[sorted_args[0]]+EQ_DELTA)
			continue;

		if (m_nominal[i])
		{
			SGVector<int32_t> simple_feats(num_vecs);
			simple_feats.fill_vector(simple_feats.vector,simple_feats.vlen,-1);

			// convert to simple values
			simple_feats[sorted_args[0]]=0;
			int32_t c=0;
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]==feats[sorted_args[j-1]])
					simple_feats[sorted_args[j]]=c;
				else
					simple_feats[sorted_args[j]]=(++c);
			}

			SGVector<float64_t> ufeats(c+1);
			ufeats[0]=feats[sorted_args[0]];
			int32_t u=0;
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]==feats[sorted_args[j-1]])
					continue;
				else
					ufeats[++u]=feats[sorted_args[j]];
			}

			// test all 2^(I-1)-1 possible division between two nodes
			int32_t num_cases=CMath::pow(2,c);
			for (int32_t k=1;k<num_cases;k++)
			{
				SGVector<float64_t> wleft(n_ulabels);
				SGVector<float64_t> wright(n_ulabels);
				wleft.zero();
				wright.zero();

				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs);
				is_left.fill_vector(is_left.vector,is_left.vlen,false);

				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(c+1);

				// fill feats_left in a unique way corresponding to the case
				for (int32_t p=0;p<c+1;p++)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (int32_t j=0;j<n_nm_vecs;j++)
				{
					is_left[sorted_args[j]]=feats_left[simple_feats[sorted_args[j]]];
					if (is_left[sorted_args[j]])
						wleft[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
					else
						wright[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
				}

				float64_t g=0;
				if (m_mode==PT_MULTICLASS)
					g=gain(wleft,wright,total_wclasses);
				else if (m_mode==PT_REGRESSION)
					g=gain(wleft,wright,total_wclasses,ulabels);
				else
					SG_ERROR("Undefined problem statement\n");

				if (g>max_gain)
				{
					best_attribute=i;
					max_gain=g;
					memcpy(is_left_final.vector,is_left.vector,is_left.vlen*sizeof(bool));
					num_missing_final=num_vecs-n_nm_vecs;

					count_left=0;
					for (int32_t l=0;l<c+1;l++)
						count_left=(feats_left[l])?count_left+1:count_left;

					count_right=c+1-count_left;

					int32_t l=0;
					int32_t r=0;
					for (int32_t w=0;w<c+1;w++)
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
			left_wclasses.zero();

			// O(N)
			// find best split for non-nominal attribute - choose threshold (z)
			float64_t z=feats[sorted_args[0]];  
			right_wclasses[simple_labels[sorted_args[0]]]-=weights[sorted_args[0]];
			left_wclasses[simple_labels[sorted_args[0]]]+=weights[sorted_args[0]];
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]<=z+EQ_DELTA)
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
					best_attribute=i;
					best_threshold=z;
					num_missing_final=num_vecs-n_nm_vecs;
				}

				z=feats[sorted_args[j]];
				if (feats[sorted_args[n_nm_vecs-1]]<=z+EQ_DELTA)
					break;

				right_wclasses[simple_labels[sorted_args[j]]]-=weights[sorted_args[j]];
				left_wclasses[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
			}
		}

		// restore total_wclasses
		while (n_nm_vecs<feats.vlen)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]+=weights[sorted_args[n_nm_vecs-1]];
			n_nm_vecs++;
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
		for (int32_t i=0;i<num_vecs;i++)
			is_left_final[i]=(mat(best_attribute,i)<=best_threshold);
	}

	return best_attribute;
}

SGVector<bool> CCARTree::surrogate_split(SGMatrix<float64_t> m,SGVector<float64_t> weights, SGVector<bool> nm_left, int32_t attr)
{
	// return vector - left/right belongingness
	SGVector<bool> ret(m.num_cols);

	// ditribute data with known attributes
	int32_t l=0;
	float64_t p_l=0.;
	float64_t total=0.;
	// stores indices of vectors with missing attribute 
	CDynamicArray<int32_t>* missing_vecs=new CDynamicArray<int32_t>();
	// stores lambda values corresponding to missing vectors - initialized all with 0
	CDynamicArray<float64_t>* association_index=new CDynamicArray<float64_t>();
	for (int32_t i=0;i<m.num_cols;i++)
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
			missing_vecs->push_back(i);
			association_index->push_back(0.);
		}
	}

	// for lambda calculation
	float64_t p_r=(total-p_l)/total;
	p_l/=total;
	float64_t p=CMath::min(p_r,p_l);

	// for each attribute (X') alternative to best split (X)
	for (int32_t i=0;i<m.num_rows;i++)
	{
		if (i==attr)
			continue;

		// find set of vectors with non-missing values for both X and X'
		CDynamicArray<int32_t>* intersect_vecs=new CDynamicArray<int32_t>();
		for (int32_t j=0;j<m.num_cols;j++)
		{
			if (!(CMath::fequals(m(i,j),MISSING,0) || CMath::fequals(m(attr,j),MISSING,0)))
				intersect_vecs->push_back(j);
		}

		if (intersect_vecs->get_num_elements()==0)
		{
			SG_UNREF(intersect_vecs);
			continue;
		}


		if (m_nominal[i])
			handle_missing_vecs_for_nominal_surrogate(m,missing_vecs,association_index,intersect_vecs,ret,weights,p,i);
		else
			handle_missing_vecs_for_continuous_surrogate(m,missing_vecs,association_index,intersect_vecs,ret,weights,p,i);

		SG_UNREF(intersect_vecs);
	}

	// if some missing attribute vectors are yet not addressed, use majority rule
	for (int32_t i=0;i<association_index->get_num_elements();i++)
	{
		if (association_index->get_element(i)==0.)
			ret[missing_vecs->get_element(i)]=(p_l>=p_r);
	}

	SG_UNREF(missing_vecs);
	SG_UNREF(association_index);
	return ret;
}

void CCARTree::handle_missing_vecs_for_continuous_surrogate(SGMatrix<float64_t> m, CDynamicArray<int32_t>* missing_vecs, 
		CDynamicArray<float64_t>* association_index, CDynamicArray<int32_t>* intersect_vecs, SGVector<bool> is_left, 
									SGVector<float64_t> weights, float64_t p, int32_t attr)
{
	// for lambda calculation - total weight of all vectors in X intersect X'
	float64_t denom=0.;
	SGVector<float64_t> feats(intersect_vecs->get_num_elements());
	for (int32_t j=0;j<intersect_vecs->get_num_elements();j++)
	{
		feats[j]=m(attr,intersect_vecs->get_element(j));
		denom+=weights[intersect_vecs->get_element(j)];
	}

	// unique feature values for X'
	int32_t num_unique=feats.unique(feats.vector,feats.vlen);


	// all possible splits for chosen attribute
	for (int32_t j=0;j<num_unique-1;j++)
	{
		float64_t z=feats[j];
		float64_t numer=0.;
		float64_t numerc=0.;
		for (int32_t k=0;k<intersect_vecs->get_num_elements();k++)
		{
			// if both go left or both go right
			if ((m(attr,intersect_vecs->get_element(k))<=z) && is_left[intersect_vecs->get_element(k)])
				numer+=weights[intersect_vecs->get_element(k)];
			else if ((m(attr,intersect_vecs->get_element(k))>z) && !is_left[intersect_vecs->get_element(k)])
				numer+=weights[intersect_vecs->get_element(k)];
			// complementary split cases - one goes left other right
			else if ((m(attr,intersect_vecs->get_element(k))<=z) && !is_left[intersect_vecs->get_element(k)])
				numerc+=weights[intersect_vecs->get_element(k)];
			else if ((m(attr,intersect_vecs->get_element(k))>z) && is_left[intersect_vecs->get_element(k)])
				numerc+=weights[intersect_vecs->get_element(k)];
		}

		float64_t lambda=0.;
		if (numer>=numerc)
			lambda=(p-(1-numer/denom))/p;
		else
			lambda=(p-(1-numerc/denom))/p;
		for (int32_t k=0;k<missing_vecs->get_num_elements();k++)
		{
			if ((lambda>association_index->get_element(k)) && 
			(!CMath::fequals(m(attr,missing_vecs->get_element(k)),MISSING,0)))
			{
				association_index->set_element(lambda,k);
				if (numer>=numerc)
					is_left[missing_vecs->get_element(k)]=(m(attr,missing_vecs->get_element(k))<=z);
				else
					is_left[missing_vecs->get_element(k)]=(m(attr,missing_vecs->get_element(k))>z);
			}
		}
	}
}

void CCARTree::handle_missing_vecs_for_nominal_surrogate(SGMatrix<float64_t> m, CDynamicArray<int32_t>* missing_vecs, 
		CDynamicArray<float64_t>* association_index, CDynamicArray<int32_t>* intersect_vecs, SGVector<bool> is_left, 
									SGVector<float64_t> weights, float64_t p, int32_t attr)
{
	// for lambda calculation - total weight of all vectors in X intersect X'
	float64_t denom=0.;
	SGVector<float64_t> feats(intersect_vecs->get_num_elements());
	for (int32_t j=0;j<intersect_vecs->get_num_elements();j++)
	{
		feats[j]=m(attr,intersect_vecs->get_element(j));
		denom+=weights[intersect_vecs->get_element(j)];
	}

	// unique feature values for X'
	int32_t num_unique=feats.unique(feats.vector,feats.vlen);

	// scan all splits for chosen alternative attribute X'
	int32_t num_cases=CMath::pow(2,(num_unique-1));
	for (int32_t j=1;j<num_cases;j++)
	{
		SGVector<bool> feats_left(num_unique);
		for (int32_t k=0;k<num_unique;k++)
			feats_left[k]=((j/CMath::pow(2,k))%(CMath::pow(2,k+1))==1);

		SGVector<bool> intersect_vecs_left(intersect_vecs->get_num_elements());
		for (int32_t k=0;k<intersect_vecs->get_num_elements();k++)
		{
			for (int32_t q=0;q<num_unique;q++)
			{
				if (feats[q]==m(attr,intersect_vecs->get_element(k)))
				{
					intersect_vecs_left[k]=feats_left[q];
					break;
				}
			}
		}

		float64_t numer=0.;
		float64_t numerc=0.;
		for (int32_t k=0;k<intersect_vecs->get_num_elements();k++)
		{
			// if both go left or both go right
			if (intersect_vecs_left[k]==is_left[intersect_vecs->get_element(k)])
				numer+=weights[intersect_vecs->get_element(k)];
			else
				numerc+=weights[intersect_vecs->get_element(k)];
		}

		// lambda for this split (2 case identical split/complementary split)
		float64_t lambda=0.;
		if (numer>=numerc)
			lambda=(p-(1-numer/denom))/p;
		else
			lambda=(p-(1-numerc/denom))/p;

		// address missing value vectors not yet addressed or addressed using worse split 
		for (int32_t k=0;k<missing_vecs->get_num_elements();k++)
		{
			if ((lambda>association_index->get_element(k)) && 
			(!CMath::fequals(m(attr,missing_vecs->get_element(k)),MISSING,0)))
			{
				association_index->set_element(lambda,k);
				// decide left/right based on which feature value the chosen data point has
				for (int32_t q=0;q<num_unique;q++)
				{
					if (feats[q]==m(attr,missing_vecs->get_element(k)))
					{
						if (numer>=numerc)
							is_left[missing_vecs->get_element(k)]=feats_left[q];
						else
							is_left[missing_vecs->get_element(k)]=!feats_left[q];

						break;
					}
				}
			}
		}
	}
}

float64_t CCARTree::gain(SGVector<float64_t> wleft, SGVector<float64_t> wright, SGVector<float64_t> wtotal,	
						SGVector<float64_t> feats)
{
	float64_t total_lweight=0;
	float64_t total_rweight=0;
	float64_t total_weight=0;

	float64_t lsd_n=least_squares_deviation(feats,wtotal,total_weight);
	float64_t lsd_l=least_squares_deviation(feats,wleft,total_lweight);
	float64_t lsd_r=least_squares_deviation(feats,wright,total_rweight);

	return lsd_n-(lsd_l*(total_lweight/total_weight))-(lsd_r*(total_rweight/total_weight));
}

float64_t CCARTree::gain(SGVector<float64_t> wleft, SGVector<float64_t> wright, SGVector<float64_t> wtotal)
{
	float64_t total_lweight=0;
	float64_t total_rweight=0;
	float64_t total_weight=0;

	float64_t gini_n=gini_impurity_index(wtotal,total_weight);
	float64_t gini_l=gini_impurity_index(wleft,total_lweight);
	float64_t gini_r=gini_impurity_index(wright,total_rweight);
	return gini_n-(gini_l*(total_lweight/total_weight))-(gini_r*(total_rweight/total_weight));
}

float64_t CCARTree::gini_impurity_index(SGVector<float64_t> weighted_lab_classes, float64_t &total_weight)
{
	total_weight=0;
	float64_t gini=0;
	for (int32_t i=0;i<weighted_lab_classes.vlen;i++)
	{
		total_weight+=weighted_lab_classes[i];
		gini+=weighted_lab_classes[i]*weighted_lab_classes[i];
	}

	gini=1.0-(gini/(total_weight*total_weight));
	return gini;
}

float64_t CCARTree::least_squares_deviation(SGVector<float64_t> feats, SGVector<float64_t> weights, float64_t &total_weight)
{
	float64_t mean=0;
	total_weight=0;
	for (int32_t i=0;i<weights.vlen;i++)
	{
		mean+=feats[i]*weights[i];
		total_weight+=weights[i];
	}

	mean/=total_weight;
	float64_t dev=0;
	for (int32_t i=0;i<weights.vlen;i++)
		dev+=weights[i]*(feats[i]-mean)*(feats[i]-mean);

	return dev/total_weight;
}

CLabels* CCARTree::apply_from_current_node(CDenseFeatures<float64_t>* feats, bnode_t* current)
{
	int32_t num_vecs=feats->get_num_vectors();
	SGVector<float64_t> labels(num_vecs);
	for (int32_t i=0;i<num_vecs;i++)
	{
		SGVector<float64_t> sample=feats->get_feature_vector(i);
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
				for (int32_t k=0;k<comp.vlen;k++)
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
	int32_t num_vecs=data->get_num_vectors();

	// divide data into V folds randomly
	SGVector<int32_t> subid(num_vecs);
	subid.random_vector(subid.vector,subid.vlen,0,folds-1);

	// for each fold subset
	CDynamicArray<float64_t>* r_cv=new CDynamicArray<float64_t>();
	CDynamicArray<float64_t>* alphak=new CDynamicArray<float64_t>();
	SGVector<int32_t> num_alphak(folds);
	for (int32_t i=0;i<folds;i++)
	{
		// for chosen fold, create subset for training parameters
		CDynamicArray<int32_t>* test_indices=new CDynamicArray<int32_t>();
		CDynamicArray<int32_t>* train_indices=new CDynamicArray<int32_t>();
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (subid[j]==i)
				test_indices->push_back(j);
			else
				train_indices->push_back(j);
		}

		if (test_indices->get_num_elements()==0 || train_indices->get_num_elements()==0)
		{
			SG_ERROR("Unfortunately you have reached the very low probability event where atleast one of "
					"the subsets in cross-validation is not represented at all. Please re-run.")
		}

		SGVector<int32_t> subset(train_indices->get_array(),train_indices->get_num_elements(),false);
		data->add_subset(subset);
		m_labels->add_subset(subset);
		SGVector<float64_t> subset_weights(train_indices->get_num_elements());
		for (int32_t j=0;j<train_indices->get_num_elements();j++)
			subset_weights[j]=m_weights[train_indices->get_element(j)];

		// train with training subset
		bnode_t* root=CARTtrain(data,subset_weights,m_labels,0);

		// prune trained tree
		CTreeMachine<CARTreeNodeData>* tmax=new CTreeMachine<CARTreeNodeData>();
		tmax->set_root(root);
		CDynamicObjectArray* pruned_trees=prune_tree(tmax);

		data->remove_subset();
		m_labels->remove_subset();
		subset=SGVector<int32_t>(test_indices->get_array(),test_indices->get_num_elements(),false);
		data->add_subset(subset);
		m_labels->add_subset(subset);
		subset_weights=SGVector<float64_t>(test_indices->get_num_elements());
		for (int32_t j=0;j<test_indices->get_num_elements();j++)
			subset_weights[j]=m_weights[test_indices->get_element(j)];

		// calculate R_CV values for each alpha_k using test subset and store them
		num_alphak[i]=m_alphas->get_num_elements();
		for (int32_t j=0;j<m_alphas->get_num_elements();j++)
		{
			alphak->push_back(m_alphas->get_element(j));
			CSGObject* jth_element=pruned_trees->get_element(j);
			bnode_t* current_root=NULL;
			if (jth_element!=NULL)
				current_root=dynamic_cast<bnode_t*>(jth_element);
			else
				SG_ERROR("%d element is NULL which should not be",j);

			CLabels* labels=apply_from_current_node(data, current_root);
			float64_t error=compute_error(labels, m_labels, subset_weights);
			r_cv->push_back(error);
			SG_UNREF(labels);
			SG_UNREF(jth_element);
		}

		data->remove_subset();
		m_labels->remove_subset();
		SG_UNREF(train_indices);
		SG_UNREF(test_indices);
		SG_UNREF(tmax);
		SG_UNREF(pruned_trees);
	}

	// prune the original T_max
	CDynamicObjectArray* pruned_trees=prune_tree(this);

	// find subtree with minimum R_cv 
	int32_t min_index=-1;
	float64_t min_r_cv=CMath::MAX_REAL_NUMBER;
	for (int32_t i=0;i<m_alphas->get_num_elements();i++)
	{
		float64_t alpha=0.;	
		if (i==m_alphas->get_num_elements()-1)
			alpha=m_alphas->get_element(i)+1;
		else
			alpha=CMath::sqrt(m_alphas->get_element(i)*m_alphas->get_element(i+1));

		float64_t rv=0.;
		int32_t base=0;
		for (int32_t j=0;j<folds;j++)
		{
			bool flag=false;
			for (int32_t k=base;k<num_alphak[j]+base-1;k++)
			{
				if (alphak->get_element(k)<=alpha && alphak->get_element(k+1)>alpha)
				{
					rv+=r_cv->get_element(k);
					flag=true;
					break;
				}
			}

			if (!flag)
				rv+=r_cv->get_element(num_alphak[j]+base-1);

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
	if (element!=NULL)
		best_tree_root=dynamic_cast<bnode_t*>(element);
	else
		SG_ERROR("%d element is NULL which should not be",min_index);

	this->set_root(best_tree_root);

	SG_UNREF(element);
	SG_UNREF(pruned_trees);	
	SG_UNREF(r_cv);
	SG_UNREF(alphak);
}

float64_t CCARTree::compute_error(CLabels* labels, CLabels* reference, SGVector<float64_t> weights)
{
	REQUIRE(labels,"input labels cannot be NULL");
	REQUIRE(reference,"reference labels cannot be NULL")

	CDenseLabels* gnd_truth=dynamic_cast<CDenseLabels*>(reference);
	CDenseLabels* result=dynamic_cast<CDenseLabels*>(labels);

	float64_t denom=weights.sum(weights);
	float64_t numer=0.;
	switch (m_mode)
	{
		case PT_MULTICLASS:
		{
			for (int32_t i=0;i<weights.vlen;i++)
			{
				if (gnd_truth->get_label(i)!=result->get_label(i))
					numer+=weights[i];
			}

			return numer/denom;
		}

		case PT_REGRESSION:
		{
			for (int32_t i=0;i<weights.vlen;i++)
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

float64_t CCARTree::find_weakest_alpha(bnode_t* node)
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
		return weak_links.min(weak_links.vector,weak_links.vlen);		
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
	m_types_set=false;
	m_weights_set=false;
	m_apply_cv_pruning=false;
	m_folds=5;
	m_alphas=new CDynamicArray<float64_t>();
	SG_REF(m_alphas);
	m_max_depth=0;
	m_min_node_size=0;
	m_label_epsilon=1e-7;

	SG_ADD(&m_nominal,"m_nominal", "feature types", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_set,"m_weights_set", "weights set", MS_NOT_AVAILABLE);
	SG_ADD(&m_types_set,"m_types_set", "feature types set", MS_NOT_AVAILABLE);
	SG_ADD(&m_apply_cv_pruning,"m_apply_cv_pruning", "apply cross validation pruning", MS_NOT_AVAILABLE);
	SG_ADD(&m_folds,"m_folds","number of subsets for cross validation", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_depth,"m_max_depth","max allowed tree depth",MS_NOT_AVAILABLE)
	SG_ADD(&m_min_node_size,"m_min_node_size","min allowed node size",MS_NOT_AVAILABLE)
	SG_ADD(&m_label_epsilon,"m_label_epsilon","epsilon for labels",MS_NOT_AVAILABLE)
}
