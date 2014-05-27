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
	SGMatrix<float64_t> mat=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_feature_matrix();
	int32_t num_feats=mat.num_rows;
	int32_t num_vecs=mat.num_cols;

	// calculate node label
	switch(m_mode)
	{
		case PT_REGRESSION:
			{
				SGVector<float64_t> lab=(dynamic_cast<CRegressionLabels*>(labels))->get_labels();
				float64_t sum=0;
				float64_t tot=0;
				for (int32_t i=0;i<lab.vlen;i++)
				{
					tot+=weights[i];
					sum+=lab[i]*weights[i];
				}
	
				node->data.node_label=sum/tot;
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
		return node;

	// case 2 : all non-dependent attributes are same
	bool flag=true;
	for (int32_t v=1;v<num_vecs;v++)
	{
		for (int32_t f=0;f<num_feats;f++)
		{
			if (mat(f,v)!=mat(f,v-1))
			{
				flag=false;
				break;
			}
		}

		if (!flag)
			break;
	}

	if (flag)
		return node;

	// choose best attribute
	float64_t max_gain=-1;
	int32_t best_attribute=-1;
	// transit_into_values for left child
	SGVector<float64_t> left;
	// transit_into_values for right child
	SGVector<float64_t> right;
	// final data distribution among children
	SGVector<bool> is_left_final;

	for (int32_t i=0;i<num_feats;i++)
	{
		SGVector<float64_t> feats(num_vecs);
		for (int32_t j=0;j<num_vecs;j++)
			feats[j]=mat(i,j);

		int32_t num_unique=feats.unique(feats.vector,feats.vlen);
		if (num_unique==1)
			continue;

		if (m_nominal[i])
		{
			// test all 2^(I-1)-1 possible division between two nodes
			int32_t num_cases=CMath::pow(2,(num_unique-1));
			for (int32_t k=1;k<num_cases;k++)
			{
				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs);
				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(num_unique);

				// fill feats_left in a unique way corresponding to the case
				for (int32_t p=0;p<num_unique;p++)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (int32_t j=0;j<num_vecs;j++)
				{
					// determine categorical value of jth vector
					int32_t index=-1;
					for (int32_t p=0;p<num_unique;p++)
					{
						if (mat(i,j)==feats[p])
						{
							index=p;
							break;
						}
					}

					// assign jth vector a child corresponding to its categorical value
					is_left[j]=feats_left[index];
				}

				float64_t g=gain(labels,weights,is_left);
				if (g>max_gain)
				{
					best_attribute=i;
					max_gain=g;
					is_left_final=is_left.clone();

					int32_t count_left=0;
					for (int32_t c=0;c<num_unique;c++)
						count_left=(feats_left[c])?count_left+1:count_left;

					left=SGVector<float64_t>(count_left);
					right=SGVector<float64_t>(num_unique-count_left);
					int32_t l=0;
					int32_t r=0;
					for (int32_t c=0;c<num_unique;c++)
					{
						if (feats_left[c])
							left[l++]=feats[c];
						else
							right[r++]=feats[c];
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
				float64_t z=feats[j];

				SGVector<bool> is_left(num_vecs);
				for (int32_t k=0;k<num_vecs;k++)
					is_left[k]=(mat(i,k)<=z);

				float64_t g=gain(labels,weights,is_left);
				if (g>max_gain)
				{
					max_gain=g;
					best_attribute=i;
					left=SGVector<float64_t>(1);
					right=SGVector<float64_t>(1);
					left[0]=z;
					right[0]=z;
					is_left_final=is_left.clone();
				}
			}
		}
	}

	int32_t count_left=0;
	for (int32_t c=0;c<num_vecs;c++)
		count_left=(is_left_final[c])?count_left+1:count_left;

	SGVector<index_t> subsetl(count_left);
	SGVector<float64_t> weightsl(count_left);
	SGVector<index_t> subsetr(num_vecs-count_left);
	SGVector<float64_t> weightsr(num_vecs-count_left);
	index_t l=0;
	index_t r=0;
	for (int32_t c=0;c<num_vecs;c++)
	{
		if (is_left_final[c])
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

	return node;
}

float64_t CCARTree::gain(CLabels* labels, SGVector<float64_t> weights, SGVector<bool> is_left)
{
	SGVector<float64_t> lab=(dynamic_cast<CDenseLabels*>(labels))->get_labels();
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
			CMulticlassLabels* labelsl=new CMulticlassLabels(lab_left);
			CMulticlassLabels* labelsr=new CMulticlassLabels(lab_right);

			float64_t gini_n=gini_impurity_index(dynamic_cast<CMulticlassLabels*>(labels),weights);
			float64_t gini_l=gini_impurity_index(labelsl,weights_left);
			float64_t gini_r=gini_impurity_index(labelsr,weights_right);

			SG_UNREF(labelsl);
			SG_UNREF(labelsr);
	
			return gini_n-(gini_l*(total_lweight/total_weight))-(gini_r*(total_rweight/total_weight));
		}

		case PT_REGRESSION:
		{
			CRegressionLabels* labelsl=new CRegressionLabels(lab_left);
			CRegressionLabels* labelsr=new CRegressionLabels(lab_right);

			float64_t lsd_n=least_squares_deviation(dynamic_cast<CRegressionLabels*>(labels),weights);
			float64_t lsd_l=least_squares_deviation(labelsl,weights_left);
			float64_t lsd_r=least_squares_deviation(labelsr,weights_right);

			SG_UNREF(labelsl);
			SG_UNREF(labelsr);

			return lsd_n-(lsd_l*(total_lweight/total_weight))-(lsd_r*(total_rweight/total_weight));	
		}

		default:
			SG_ERROR("mode should be either PT_MULTICLASS or PT_REGRESSION\n");
	}

	return -1.0;
}

float64_t CCARTree::gini_impurity_index(CMulticlassLabels* labels, SGVector<float64_t> weights)
{
	if (weights.vlen==1)
		return 1.0;

	SGVector<float64_t> lab=labels->get_labels();
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

float64_t CCARTree::least_squares_deviation(CRegressionLabels* labels, SGVector<float64_t> weights)
{
	SGVector<float64_t> lab=labels->get_labels();
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
