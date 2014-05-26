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

const float64_t CCARTree::MISSING=CMath::NOT_A_NUMBER;

CCARTree::CCARTree()
: CTreeMachine<CARTreeNodeData>()
{
	init();
}

CCARTree::~CCARTree()
{
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

CDynamicObjectArray* CCARTree::prune_tree()
{
	CDynamicObjectArray* trees=new CDynamicObjectArray();
	SG_UNREF(m_alphas);
	m_alphas=new CDynamicArray<float64_t>();
	SG_REF(m_alphas);

	// base tree alpha_k=0
	m_alphas->push_back(0);
	CTreeMachine<CARTreeNodeData>* t1=this->clone_tree();
	bnode_t* t1_root=dynamic_cast<bnode_t*>(t1->get_root());
	form_t1(t1_root);
	trees->push_back(t1);
	while(t1_root->data.num_leaves>1)
	{
		CTreeMachine<CARTreeNodeData>* t2=t1->clone_tree();
		SG_REF(t2);

		bnode_t* t2_root=dynamic_cast<bnode_t*>(t2->get_root());
		float64_t a_k=find_weakest_alpha(t2_root);
		m_alphas->push_back(a_k);
		cut_weakest_link(t2_root,a_k);
		trees->push_back(t2);

		SG_UNREF(t1);
		SG_UNREF(t2_root);
		t1=t2;
	}

	SG_UNREF(t1);
	SG_UNREF(t1_root);
	return trees;
}

SGVector<float64_t> CCARTree::get_alphas() const
{
	int32_t num=m_alphas->get_num_elements();
	return SGVector<float64_t>(m_alphas->get_array(),num);
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
	bnode_t* left=node->left();
	bnode_t* right=node->right();

	if (left!=NULL && right!=NULL)
	{
		form_t1(left);
		form_t1(right);
		node->data.num_leaves=left->data.num_leaves+right->data.num_leaves;
		node->data.weight_minus_branch=left->data.weight_minus_branch+right->data.weight_minus_branch;
		if (node->data.weight_minus_node==node->data.weight_minus_branch);
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

	set_root(CARTtrain(data,m_weights,m_labels));

	return true;
}

CBinaryTreeMachineNode<CARTreeNodeData>* CCARTree::CARTtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels)
{

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
				float64_t tot=0;
				for (int32_t i=0;i<labels_vec.vlen;i++)
				{
					tot+=weights[i];
					sum+=labels_vec[i]*weights[i];
				}
	
				node->data.node_label=sum/tot;

				node->data.total_weight=tot;

				// lsd*total_weight=sum_of_squared_deviation
				node->data.weight_minus_node=tot*least_squares_deviation(labels_vec,weights);
				break;
			}
		case PT_MULTICLASS:
			{
				SGVector<float64_t> lab=(dynamic_cast<CMulticlassLabels*>(labels))->get_labels_copy();	
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
	// case 1 : all labels same
	SGVector<float64_t> lab=(dynamic_cast<CDenseLabels*>(labels))->get_labels_copy();
	int32_t unique=lab.unique(lab.vector,lab.vlen);
	if (unique==1)
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	// case 2 : all non-dependent attributes (not MISSING) are same
	bool flag=true;
	for (int32_t v=1;v<num_vecs;v++)
	{
		for (int32_t f=0;f<num_feats;f++)
		{
			if (!(CMath::fequals(mat(f,v),MISSING,0)) && !(CMath::fequals(mat(f,v-1),MISSING,0)))
			{
				if (mat(f,v)!=mat(f,v-1))
				{
					flag=false;
					break;
				}
			}
		}

		if (!flag)
			break;
	}

	if (flag)
	{
		node->data.num_leaves=1;
		node->data.weight_minus_branch=node->data.weight_minus_node;
		return node;
	}

	// choose best attribute
	float64_t max_gain=-1;
	int32_t best_attribute=-1;
	// transit_into_values for left child
	SGVector<float64_t> left;
	// transit_into_values for right child
	SGVector<float64_t> right;
	// final data distribution among children
	SGVector<bool> is_left_final;
	int32_t num_missing_final=0;

	for (int32_t i=0;i<num_feats;i++)
	{
		// find number of missing data points for chosen attribute
		int32_t num_missing=0;
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (CMath::fequals(mat(i,j),MISSING,0))
				num_missing++;
		}

		// assimilate non-missing features, corresponding weights and labels
		SGVector<float64_t> non_missing_feats(num_vecs-num_missing);
		SGVector<float64_t> non_missing_weights(num_vecs-num_missing);
		SGVector<float64_t> non_missing_labels(num_vecs-num_missing);
		int32_t c=0;
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (!CMath::fequals(mat(i,j),MISSING,0))
			{
				non_missing_feats[c]=mat(i,j);
				non_missing_weights[c]=weights[j];
				non_missing_labels[c++]=labels_vec[j];
			}
		}

		// get unique feature values
		SGVector<float64_t> nm_feats_copy=non_missing_feats.clone();
		int32_t num_unique=nm_feats_copy.unique(nm_feats_copy.vector,nm_feats_copy.vlen);
		// if only one unique value - it cannot be used to split
		if (num_unique==1)
			continue;

		if (m_nominal[i])
		{
			// test all 2^(I-1)-1 possible division between two nodes
			int32_t num_cases=CMath::pow(2,(num_unique-1));
			for (int32_t k=1;k<num_cases;k++)
			{
				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs-num_missing);
				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(num_unique);

				// fill feats_left in a unique way corresponding to the case
				for (int32_t p=0;p<num_unique;p++)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (int32_t j=0;j<num_vecs-num_missing;j++)
				{
					// determine categorical value of jth vector
					int32_t index=-1;
					for (int32_t p=0;p<num_unique;p++)
					{
						if (non_missing_feats[j]==nm_feats_copy[p])
						{
							index=p;
							break;
						}
					}

					// assign jth vector a child corresponding to its categorical value
					is_left[j]=feats_left[index];
				}

				float64_t g=gain(non_missing_labels,non_missing_weights,is_left);
				if (g>max_gain)
				{
					best_attribute=i;
					max_gain=g;
					is_left_final=is_left.clone();
					num_missing_final=num_missing;

					int32_t count_left=0;
					for (int32_t l=0;l<num_unique;l++)
						count_left=(feats_left[l])?count_left+1:count_left;

					left=SGVector<float64_t>(count_left);
					right=SGVector<float64_t>(num_unique-count_left);
					int32_t l=0;
					int32_t r=0;
					for (int32_t w=0;w<num_unique;w++)
					{
						if (feats_left[w])
							left[l++]=nm_feats_copy[w];
						else
							right[r++]=nm_feats_copy[w];
					}
				}
			}
		}
		else
		{
			// find best split for non-nominal attribute - choose threshold (z)  
			for (int32_t j=0;j<num_unique-1;j++)
			{
				// threshold
				float64_t z=nm_feats_copy[j];

				SGVector<bool> is_left(num_vecs-num_missing);
				for (int32_t k=0;k<num_vecs-num_missing;k++)
					is_left[k]=(non_missing_feats[k]<=z);

				float64_t g=gain(non_missing_labels,non_missing_weights,is_left);
				if (g>max_gain)
				{
					max_gain=g;
					best_attribute=i;
					num_missing_final=num_missing;
					left=SGVector<float64_t>(1);
					right=SGVector<float64_t>(1);
					left[0]=z;
					right[0]=z;
					is_left_final=is_left.clone();
				}
			}
		}
	}

	SGVector<bool> is_left(num_vecs);
	if (num_missing_final>0)
		is_left=surrogate_split(mat,weights,is_left_final,best_attribute);
	else
		is_left=is_left_final;

	int32_t count_left=0;
	for (int32_t c=0;c<num_vecs;c++)
		count_left=(is_left[c])?count_left+1:count_left;

	SGVector<index_t> subsetl(count_left);
	SGVector<float64_t> weightsl(count_left);
	SGVector<index_t> subsetr(num_vecs-count_left);
	SGVector<float64_t> weightsr(num_vecs-count_left);
	index_t l=0;
	index_t r=0;
	for (int32_t c=0;c<num_vecs;c++)
	{
		if (is_left[c])
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
	bnode_t* left_child=CARTtrain(data,weightsl,labels);
	data->remove_subset();
	labels->remove_subset();

	// right child
	data->add_subset(subsetr);
	labels->add_subset(subsetr);
	bnode_t* right_child=CARTtrain(data,weightsr,labels);
	data->remove_subset();
	labels->remove_subset();

	// set node parameters
	node->data.attribute_id=best_attribute;
	node->left(left_child);
	node->right(right_child);
	left_child->data.transit_into_values=left;
	right_child->data.transit_into_values=right;
	node->data.num_leaves=left_child->data.num_leaves+right_child->data.num_leaves;
	node->data.weight_minus_branch=left_child->data.weight_minus_branch+right_child->data.weight_minus_branch;

	return node;
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
							is_left[missing_vecs->get_element(k)]=~feats_left[q];

						break;
					}
				}
			}
		}
	}
}

float64_t CCARTree::gain(SGVector<float64_t> lab, SGVector<float64_t> weights, SGVector<bool> is_left)
{
	float64_t total_weight=weights.sum(weights);

	int32_t num_left=0;
	for (int32_t i=0;i<is_left.vlen;i++)
		num_left=(is_left[i])?num_left+1:num_left;

	SGVector<float64_t> weights_left(num_left);
	SGVector<float64_t> weights_right(is_left.vlen-num_left);
	SGVector<float64_t> lab_left(num_left);
	SGVector<float64_t> lab_right(is_left.vlen-num_left);
	float64_t total_lweight=0;
	float64_t total_rweight=0;
	int32_t l=0;
	int32_t r=0; 
	for (int32_t i=0;i<is_left.vlen;i++)
	{
		if (is_left[i])
		{
			weights_left[l]=weights[i];
			lab_left[l]=lab[i];
			total_lweight+=weights_left[l];
			l++;
		}
		else
		{
			weights_right[r]=weights[i];
			lab_right[r]=lab[i];
			total_rweight+=weights_right[r];
			r++;
		}
	}

	switch(m_mode)
	{
		case PT_MULTICLASS:
		{
			float64_t gini_n=gini_impurity_index(lab,weights);
			float64_t gini_l=gini_impurity_index(lab_left,weights_left);
			float64_t gini_r=gini_impurity_index(lab_right,weights_right);

			return gini_n-(gini_l*(total_lweight/total_weight))-(gini_r*(total_rweight/total_weight));
		}

		case PT_REGRESSION:
		{
			float64_t lsd_n=least_squares_deviation(lab,weights);
			float64_t lsd_l=least_squares_deviation(lab_left,weights_left);
			float64_t lsd_r=least_squares_deviation(lab_right,weights_right);

			return lsd_n-(lsd_l*(total_lweight/total_weight))-(lsd_r*(total_rweight/total_weight));	
		}

		default:
			SG_ERROR("mode should be either PT_MULTICLASS or PT_REGRESSION\n");
	}

	return -1.0;
}

float64_t CCARTree::gini_impurity_index(SGVector<float64_t> lab, SGVector<float64_t> weights)
{
	if (weights.vlen==1)
		return 1.0;

	float64_t total_weight=weights.sum(weights);
	SGVector<index_t> sorted_args=lab.argsort();
	float64_t gini=1;
	float64_t minus=weights[sorted_args[0]];
	for (int32_t i=1;i<sorted_args.vlen;i++)
	{
		if (lab[sorted_args[i]]==lab[sorted_args[i-1]])
		{
			minus+=weights[sorted_args[i]];
		}
		else
		{
			gini-=(minus/total_weight)*(minus/total_weight);
			minus=weights[sorted_args[i]];
		}
	}

	gini-=(minus/total_weight)*(minus/total_weight);
	return gini;
}

float64_t CCARTree::least_squares_deviation(SGVector<float64_t> lab, SGVector<float64_t> weights)
{
	float64_t mean=0;
	float64_t total_weight=0;
	for (int32_t i=0;i<lab.vlen;i++)
	{
		mean+=lab[i]*weights[i];
		total_weight+=weights[i];
	}

	mean/=total_weight;
	float64_t dev=0;
	for (int32_t i=0;i<lab.vlen;i++)
		dev+=weights[i]*(lab[i]-mean)*(lab[i]-mean);

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
		while(node->data.attribute_id!=-1)
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

void CCARTree::init()
{
	m_nominal=SGVector<bool>();
	m_weights=SGVector<float64_t>();
	m_mode=PT_MULTICLASS;
	m_types_set=false;
	m_weights_set=false;

	SG_ADD(&m_nominal,"m_nominal", "feature types", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_set,"m_weights_set", "weights set", MS_NOT_AVAILABLE);
	SG_ADD(&m_types_set,"m_types_set", "feature types set", MS_NOT_AVAILABLE);
}
