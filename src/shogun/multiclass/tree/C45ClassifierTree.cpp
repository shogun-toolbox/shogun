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
#include <shogun/multiclass/tree/C45ClassifierTree.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

const float64_t CC45ClassifierTree::MISSING=CMath::NOT_A_NUMBER; 

CC45ClassifierTree::CC45ClassifierTree()
: CTreeMachine<C45TreeNodeData>()
{
	init();
}

CC45ClassifierTree::~CC45ClassifierTree()
{
}

CMulticlassLabels* CC45ClassifierTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	// apply multiclass starting from root
	node_t* current=get_root();
	CMulticlassLabels* ret=apply_multiclass_from_current_node(dynamic_cast<CDenseFeatures<float64_t>*>(data), current, true);

	SG_UNREF(current);
	return ret;
}

bool CC45ClassifierTree::prune_tree(CDenseFeatures<float64_t>* validation_data, CMulticlassLabels* validation_labels, float64_t epsilon)
{
	node_t* current=get_root();
	prune_tree_from_current_node(validation_data,validation_labels,current,epsilon);

	SG_UNREF(current);
	return true;
}

SGVector<float64_t> CC45ClassifierTree::get_certainty_vector() const
{
	return m_certainty;
}

void CC45ClassifierTree::set_weights(SGVector<float64_t> w)
{
	m_weights=w;
	m_weights_set=true;
}

SGVector<float64_t> CC45ClassifierTree::get_weights() const
{
	return m_weights;
}

void CC45ClassifierTree::clear_weights()
{
	m_weights=SGVector<float64_t>();
	m_weights_set=false;
}

void CC45ClassifierTree::set_feature_types(SGVector<bool> ft)
{
	m_nominal=ft;
	m_types_set=true;
}

SGVector<bool> CC45ClassifierTree::get_feature_types() const
{
	return m_nominal;
}

void CC45ClassifierTree::clear_feature_types()
{
	m_nominal=SGVector<bool>();
	m_types_set=false;
}

bool CC45ClassifierTree::train_machine(CFeatures* data)
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

	SGVector<int32_t> feature_ids(num_features);
	feature_ids.range_fill();

	set_root(C45train(data, m_weights, dynamic_cast<CMulticlassLabels*>(m_labels), feature_ids, 0));

	return true;
}

CTreeMachineNode<C45TreeNodeData>* CC45ClassifierTree::C45train(CFeatures* data, SGVector<float64_t> weights, 
	CMulticlassLabels* class_labels, SGVector<int32_t> feature_id_vector, int32_t level)
{
	node_t* node=new node_t();
	CDenseFeatures<float64_t>* feats=dynamic_cast<CDenseFeatures<float64_t>*>(data);
	int32_t num_vecs=feats->get_num_vectors();

	// set class_label for the node as the mode of occurring multiclass labels
	SGVector<float64_t> labels=class_labels->get_labels_copy();
	labels.qsort();

	int32_t most_label=labels[0];
	int32_t most_weight=weights[0];
	int32_t weight=weights[0];

	for (int32_t i=1; i<labels.vlen; i++)
	{
		if (labels[i]==labels[i-1])
		{
			weight+=weights[i];
		}
		else if (weight>most_weight)
		{
			most_weight=weight;
			most_label=labels[i-1];
			weight=weights[i];
		}
		else
		{
			weight=weights[i];
		}	
	}

	if (weight>most_weight)
	{
		most_weight=weight;
		most_label=labels[labels.vlen-1];
	}

	node->data.class_label=most_label;
	node->data.total_weight=weights.sum(weights.vector,weights.vlen);
	node->data.weight_minus=0.0;
	for (int32_t i=0;i<labels.vlen;i++)
	{
		if (class_labels->get_label(i)!=most_label)
			node->data.weight_minus+=weights[i];
	}

	// if all samples belong to the same class
	if (class_labels->get_unique_labels().size()==1)
		return node;

	// if no feature is left
	if (feature_id_vector.vlen==0)
		return node;

	// else get the feature with the highest informational gain. threshold is used for continuous features only.
	float64_t max=0;
	int32_t best_feature_index=-1;
	float64_t threshold=0.;
	for (int32_t i=0; i<feats->get_num_features(); i++)
	{
		if (m_nominal[feature_id_vector[i]])
		{
			float64_t gain=informational_gain_attribute(i,feats,weights,class_labels);
			if (gain>=max)
			{
				max=gain;
				best_feature_index=i;
			}
		}
		else
		{
			SGVector<float64_t> feature_values(num_vecs);
			float64_t max_value=CMath::MIN_REAL_NUMBER;
			for (int32_t k=0; k<num_vecs; k++)
			{
				feature_values[k]=(feats->get_feature_vector(k))[i];

				if (!CMath::fequals(feature_values[k],MISSING,0) && feature_values[k]>max_value)
					max_value=feature_values[k];
			}

			for (int32_t k=0;k<num_vecs;k++)
			{
				if (feature_values[k]!=max_value && !CMath::fequals(feature_values[k],MISSING,0))
				{
					// form temporary dense features to calculate gain (continuous->nominal conversion)
					float64_t z=feature_values[k];
					SGMatrix<float64_t> temp_feat_mat=SGMatrix<float64_t>(1,num_vecs);
					for (int32_t l=0;l<num_vecs;l++)
					{
						if (CMath::fequals(feature_values[l],MISSING,0))
							temp_feat_mat(0,l)=MISSING;
						else if (feature_values[l]<=z)
							temp_feat_mat(0,l)=0.;
						else
							temp_feat_mat(0,l)=1.;
					}

					CDenseFeatures<float64_t>* temp_feats=new CDenseFeatures<float64_t>(temp_feat_mat); 
					float64_t gain=informational_gain_attribute(0,temp_feats,weights,class_labels);
					if (gain>max)
					{
						threshold=z;
						max=gain;
						best_feature_index=i;
					}

					SG_UNREF(temp_feats);
				}
			}
		}
	}

	// feature cache for data restoration if feature is continuous 
	SGVector<float64_t> feature_cache(num_vecs);

	// if continuous attribute - split feature values about threshold
	if (!m_nominal[feature_id_vector[best_feature_index]])
	{
		// convert continuous feature to nominal. Store cache for restoration
		SGMatrix<float64_t> modified_feat_mat=feats->get_feature_matrix();
		for(int32_t p=0;p<num_vecs;p++)
		{
			feature_cache[p]=modified_feat_mat(best_feature_index,p);
			if (CMath::fequals(modified_feat_mat(best_feature_index,p),MISSING,0))
				modified_feat_mat(best_feature_index,p)=MISSING;
			else if (modified_feat_mat(best_feature_index,p)<=threshold)
				modified_feat_mat(best_feature_index,p)=0.;
			else
				modified_feat_mat(best_feature_index,p)=1.;
		}

		feats->set_feature_matrix(modified_feat_mat);
	}

	// get feature values for the best feature chosen - shorthand for the features values of the best feature chosen
	SGVector<float64_t> best_feature_values(num_vecs);
	for (int32_t i=0; i<num_vecs; i++)
		best_feature_values[i]=(feats->get_feature_vector(i))[best_feature_index];

	// prepare vector of unique feature values excluding MISSING , also calculate total weight associated with missing attributes
	int32_t num_missing=0;
	float64_t weight_missing=0.;
	for (int32_t j=0;j<num_vecs;j++)
	{
		if (CMath::fequals(best_feature_values[j],MISSING,0))
		{
			num_missing++;
			weight_missing+=weights[j];
		}
	}

	SGVector<float64_t> best_features_unique(num_vecs-num_missing);
	int32_t index=0;
	for (int32_t j=0;j<num_vecs;j++)
	{
		if (!CMath::fequals(best_feature_values[j],MISSING,0))
			best_features_unique[index++]=best_feature_values[j];
	}

	int32_t uniques_num=best_features_unique.unique(best_features_unique.vector,best_features_unique.vlen);

	// create child node for each unique value
	for (int32_t i=0; i<uniques_num; i++)
	{
		//compute the number of vectors with active attribute value
		int32_t num_cols=0;
		float64_t active_feature_value=best_features_unique[i];

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (active_feature_value==best_feature_values[j] || CMath::fequals(best_feature_values[j],MISSING,0))
				num_cols++;
		}

		SGMatrix<float64_t> mat(feats->get_num_features()-1, num_cols);	
		SGVector<float64_t> new_labels_vector(num_cols);
		SGVector<float64_t> new_weights(num_cols);

		int32_t cnt=0;
		//choose the samples that have the active feature value
		for (int32_t j=0; j<num_vecs; j++)
		{
			SGVector<float64_t> sample=feats->get_feature_vector(j);
			if (active_feature_value==sample[best_feature_index] || CMath::fequals(sample[best_feature_index],MISSING,0))
			{
				int32_t idx=-1;
				for (int32_t k=0; k<sample.size(); k++)
				{
					if (k!=best_feature_index)			
						mat(++idx, cnt) = sample[k];
				}
	
				new_labels_vector[cnt]=class_labels->get_labels()[j];
				if (!CMath::fequals(sample[best_feature_index],MISSING,0))
					new_weights[cnt]=weights[j];
				else
					new_weights[cnt]=0.;

				cnt++;
			}
		}

		// rectify weights of data points with missing attributes (set zero previously)
		float64_t numer=new_weights.sum(new_weights.vector,new_weights.vlen);
		float64_t rec_weight=numer/(node->data.total_weight-weight_missing);
		cnt=0;
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (CMath::fequals(best_feature_values[j],MISSING,0))
				new_weights[cnt++]=rec_weight;
			else if (best_feature_values[j]==active_feature_value)
				cnt++;
		}

		//remove the best_attribute from the remaining attributes index vector
		SGVector<int32_t> new_feature_id_vector(feature_id_vector.vlen-1);		
		cnt=-1;
		for (int32_t j=0;j<feature_id_vector.vlen;j++)
		{
			if (j!=best_feature_index)
				new_feature_id_vector[++cnt]=feature_id_vector[j];		
		}

		// new data & label for child node 
		CMulticlassLabels* new_class_labels=new CMulticlassLabels(new_labels_vector);
		CDenseFeatures<float64_t>* new_data=new CDenseFeatures<float64_t>(mat);

		// recursion over child nodes	
		node_t* child=C45train(new_data,new_weights,new_class_labels,new_feature_id_vector,level+1);
		node->data.attribute_id=feature_id_vector[best_feature_index];
		if (m_nominal[feature_id_vector[best_feature_index]])
			child->data.transit_if_feature_value=active_feature_value;
		else
			child->data.transit_if_feature_value=threshold;

		node->add_child(child);

		SG_UNREF(new_class_labels);
		SG_UNREF(new_data);
	}

	// if continuous attribute - restoration required
	if (!m_nominal[feature_id_vector[best_feature_index]])
	{
		// restore data matrix
		SGMatrix<float64_t> feat_mat=feats->get_feature_matrix();
		for(int32_t p=0;p<num_vecs;p++)
			feat_mat(best_feature_index,p)=feature_cache[p];

		feats->set_feature_matrix(feat_mat);
	}

	return node;
}

void CC45ClassifierTree::prune_tree_from_current_node(CDenseFeatures<float64_t>* feats,
		CMulticlassLabels* gnd_truth, node_t* current, float64_t epsilon)
{
	// if leaf node then skip pruning
	if (current->data.attribute_id==-1)
		return;

	SGMatrix<float64_t> feature_matrix=feats->get_feature_matrix();
	CDynamicObjectArray* children=current->get_children();

	if (m_nominal[current->data.attribute_id])
	{
		for (int32_t i=0; i<children->get_num_elements(); i++)
		{
			// count number of feature vectors which transit into the child
			int32_t count=0;
			node_t* child=dynamic_cast<node_t*>(children->get_element(i));

			for (int32_t j=0; j<feature_matrix.num_cols; j++)
			{
				float64_t child_transit=child->data.transit_if_feature_value;

				if (child_transit==feature_matrix(current->data.attribute_id,j))
					count++;
			}

			if (count==0)
				continue;

			// form new subset of features and labels
			SGVector<index_t> subset=SGVector<index_t>(count);
			int32_t k=0;
	
			for (int32_t j=0; j<feature_matrix.num_cols;j++)
			{
				float64_t child_transit=child->data.transit_if_feature_value;
	
				if (child_transit==feature_matrix(current->data.attribute_id,j))
				{
					subset[k]=(index_t) j;
					k++;
				}
			}
	
			feats->add_subset(subset);
			gnd_truth->add_subset(subset);
	
			// prune the child subtree
			prune_tree_from_current_node(feats,gnd_truth,child,epsilon);
	
			feats->remove_subset();
			gnd_truth->remove_subset();
	
			SG_UNREF(child);
		}
	}
	else
	{
		REQUIRE(children->get_num_elements()==2,"The chosen attribute in current node is nominal. Expected number of"
					" children is 2 but current node has %d children.",children->get_num_elements())

		node_t* left_child=dynamic_cast<node_t*>(children->get_element(0));
		node_t* right_child=dynamic_cast<node_t*>(children->get_element(1));

		int32_t count_left=0;
		for (int32_t k=0;k<feature_matrix.num_cols;k++)
		{
			if (feature_matrix(current->data.attribute_id,k)<=left_child->data.transit_if_feature_value)
				count_left++;
		}

		SGVector<int32_t> left_subset(count_left);
		SGVector<int32_t> right_subset(feature_matrix.num_cols-count_left);
		int32_t l=0;
		int32_t r=0;
		for (int32_t k=0;k<feature_matrix.num_cols;k++)
		{
			if (feature_matrix(current->data.attribute_id,k)<=left_child->data.transit_if_feature_value)
				left_subset[l++]=k;
			else
				right_subset[r++]=k;
		}

		// count_left is 0 if entire validation data in current node moves to only right child
		if (count_left>0)
		{
			feats->add_subset(left_subset);
			gnd_truth->add_subset(left_subset);
			// prune the left child subtree
			prune_tree_from_current_node(feats,gnd_truth,left_child,epsilon);
			feats->remove_subset();
			gnd_truth->remove_subset();
		}

		// count_left is equal to num_cols if entire validation data in current node moves only to left child 
		if (count_left<feature_matrix.num_cols)
		{
			feats->add_subset(right_subset);
			gnd_truth->add_subset(right_subset);
			// prune the right child subtree
			prune_tree_from_current_node(feats,gnd_truth,right_child,epsilon);
			feats->remove_subset();
			gnd_truth->remove_subset();
		}

		SG_UNREF(left_child);
		SG_UNREF(right_child);
	}

	SG_UNREF(children);

	CMulticlassLabels* predicted_unpruned=apply_multiclass_from_current_node(feats, current);
	SGVector<float64_t> pruned_labels=SGVector<float64_t>(feature_matrix.num_cols);
	for (int32_t i=0; i<feature_matrix.num_cols; i++)
		pruned_labels[i]=current->data.class_label;

	CMulticlassLabels* predicted_pruned=new CMulticlassLabels(pruned_labels);

	CMulticlassAccuracy* accuracy=new CMulticlassAccuracy();
	float64_t unpruned_accuracy=accuracy->evaluate(predicted_unpruned, gnd_truth);
	float64_t pruned_accuracy=accuracy->evaluate(predicted_pruned, gnd_truth);
	
	if (unpruned_accuracy<pruned_accuracy+epsilon)
	{
		CDynamicObjectArray* null_children=new CDynamicObjectArray();
		current->set_children(null_children);
		SG_UNREF(null_children);
	}

	SG_UNREF(accuracy);
	SG_UNREF(predicted_pruned);
	SG_UNREF(predicted_unpruned);
}

float64_t CC45ClassifierTree::informational_gain_attribute(int32_t attr_no, CFeatures* data, 
				SGVector<float64_t> weights, CMulticlassLabels* class_labels)
{
	REQUIRE(data,"Data required for information gain calculation\n")
	REQUIRE(data->get_feature_class()==C_DENSE,
		"Dense data required for information gain calculation\n")

	float64_t gain=0;
	CDenseFeatures<float64_t>* feats=dynamic_cast<CDenseFeatures<float64_t>*>(data);
	int32_t num_vecs=feats->get_num_vectors();
	SGVector<float64_t> gain_attribute_values;
	SGVector<float64_t> gain_weights=weights;
	CMulticlassLabels* gain_labels=class_labels;

	int32_t num_missing=0;
	for (int32_t i=0;i<num_vecs;i++)
	{
		if (CMath::fequals((feats->get_feature_vector(i))[attr_no],MISSING,0))
			num_missing++;
	}

	if (num_missing==0)
	{
		gain_attribute_values=SGVector<float64_t>(num_vecs);
		for (int32_t i=0; i<num_vecs; i++)
			gain_attribute_values[i]=(feats->get_feature_vector(i))[attr_no];
	}
	else
	{
		gain_attribute_values=SGVector<float64_t>(num_vecs-num_missing);
		gain_weights=SGVector<float64_t>(num_vecs-num_missing);
		SGVector<float64_t> label_vector(num_vecs-num_missing);
		int32_t index=0;
		for (int32_t i=0; i<num_vecs; i++)
		{
			if (!CMath::fequals((feats->get_feature_vector(i))[attr_no],MISSING,0))
			{
				gain_attribute_values[index]=(feats->get_feature_vector(i))[attr_no];
				gain_weights[index]=weights[i];
				label_vector[index++]=class_labels->get_label(i);
			}
		}

		num_vecs-=num_missing;
		gain_labels=new CMulticlassLabels(label_vector);
	}

	float64_t total_weight=gain_weights.sum(gain_weights.vector,gain_weights.vlen);

	SGVector<float64_t> attr_val_unique=gain_attribute_values.clone();
	int32_t uniques_num=attr_val_unique.unique(attr_val_unique.vector,attr_val_unique.vlen);

	for (int32_t i=0; i<uniques_num; i++)
	{
		//calculate class entropy for the specific attribute_value
		int32_t attr_count=0;
		float64_t weight_count=0.;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (gain_attribute_values[j]==attr_val_unique[i])
			{
				weight_count+=gain_weights[j];
				attr_count++;
			}
		}

		SGVector<float64_t> sub_class(attr_count);
		SGVector<float64_t> sub_weights(attr_count);
		int32_t count=0;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (gain_attribute_values[j]==attr_val_unique[i])
			{
				sub_weights[count]=gain_weights[j];
				sub_class[count++]=gain_labels->get_label(j);
			}
		}
		
		CMulticlassLabels* sub_labels=new CMulticlassLabels(sub_class);
		float64_t sub_entropy=entropy(sub_labels,sub_weights);
		gain += sub_entropy*weight_count/total_weight;

		SG_UNREF(sub_labels);
	}

	float64_t data_entropy=entropy(gain_labels,gain_weights);
	gain = data_entropy-gain;

	if (num_missing!=0)
	{
		gain*=(num_vecs-0.f)/(num_vecs+num_missing-0.f);
		SG_UNREF(gain_labels);
	}

	return gain;
}

float64_t CC45ClassifierTree::entropy(CMulticlassLabels* labels, SGVector<float64_t> weights)
{
	SGVector<float64_t> log_ratios(labels->get_unique_labels().size());
	float64_t total_weight=weights.sum(weights.vector,weights.vlen);

	for (int32_t i=0;i<labels->get_unique_labels().size();i++)
	{
		int32_t count=0;
		float64_t weight_count=0.;
		for (int32_t j=0;j<labels->get_num_labels();j++)
		{
			if (labels->get_unique_labels()[i]==labels->get_label(j))
			{
				weight_count+=weights[j];
				count++;
			}
		}

		log_ratios[i]=weight_count/total_weight;
		log_ratios[i]=CMath::log(log_ratios[i]);			
	}

	return CStatistics::entropy(log_ratios.vector,log_ratios.vlen);
}

CMulticlassLabels* CC45ClassifierTree::apply_multiclass_from_current_node(CDenseFeatures<float64_t>* feats,
									node_t* current, bool set_certainty)
{
	int32_t num_vecs=feats->get_num_vectors();
	SGVector<float64_t> labels(num_vecs);
	if (set_certainty)
		m_certainty=SGVector<float64_t>(num_vecs);

	// classify vectors in feature matrix taking one at a time
	for (int32_t i=0; i<num_vecs; i++)
	{
		// choose the current subtree as the entry point
		SGVector<float64_t> sample=feats->get_feature_vector(i);
		node_t* node=current;
		SG_REF(node);
		CDynamicObjectArray* children=node->get_children();

		// traverse the subtree until leaf node is reached
		while (children->get_num_elements())
		{
			bool flag=false;
			// if nominal attribute check for equality
			if (m_nominal[node->data.attribute_id])
			{
				for (int32_t j=0; j<children->get_num_elements(); j++)
				{
					node_t* child=dynamic_cast<node_t*>(children->get_element(j));
					if (child->data.transit_if_feature_value==sample[node->data.attribute_id])
					{
						flag=true;
	
						SG_UNREF(node);
						node=child;
	
						SG_UNREF(children);
						children=node->get_children();

						break;
					}

					SG_UNREF(child);
				}

				if (!flag)
					break;
			}
			// if not nominal attribute check if greater or less than threshold
			else
			{
				node_t* left_child=dynamic_cast<node_t*>(children->get_element(0));
				node_t* right_child=dynamic_cast<node_t*>(children->get_element(1));
				if (left_child->data.transit_if_feature_value>=sample[node->data.attribute_id])
				{
					SG_UNREF(node);
					node=left_child;
					SG_REF(left_child)

					SG_UNREF(children);
					children=node->get_children();
				}
				else
				{
					SG_UNREF(node);
					node=right_child;
					SG_REF(right_child)

					SG_UNREF(children);
					children=node->get_children();
				}

				SG_UNREF(left_child);
				SG_UNREF(right_child);
			}
		}

		// class_label of leaf node is the class to which chosen vector belongs
		labels[i]=node->data.class_label;

		if (set_certainty)
			m_certainty[i]=(node->data.total_weight-node->data.weight_minus)/node->data.total_weight;

		SG_UNREF(node);
		SG_UNREF(children);
	}

	CMulticlassLabels* ret=new CMulticlassLabels(labels);
	return ret;
}

void CC45ClassifierTree::init()
{
	m_nominal=SGVector<bool>();
	m_weights=SGVector<float64_t>();
	m_certainty=SGVector<float64_t>();
	m_types_set=false;
	m_weights_set=false;

	SG_ADD(&m_nominal,"m_nominal", "feature types", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_certainty,"m_certainty", "certainty", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_set,"m_weights_set", "weights set", MS_NOT_AVAILABLE);
	SG_ADD(&m_types_set,"m_types_set", "feature types set", MS_NOT_AVAILABLE);
}

