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

#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/lib/View.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/multiclass/tree/C45ClassifierTree.h>

using namespace shogun;

const float64_t C45ClassifierTree::MISSING=Math::NOT_A_NUMBER;

C45ClassifierTree::C45ClassifierTree()
: TreeMachine<C45TreeNodeData>()
{
	init();
}

C45ClassifierTree::~C45ClassifierTree()
{
}

std::shared_ptr<MulticlassLabels> C45ClassifierTree::apply_multiclass(std::shared_ptr<Features> data)
{
	require(data, "Data required for classification in apply_multiclass");

	// apply multiclass starting from root
	auto current=get_root();
	return apply_multiclass_from_current_node(data->as<DenseFeatures<float64_t>>(), current, true);
}

void C45ClassifierTree::prune_tree(const std::shared_ptr<Features>& validation_data, const std::shared_ptr<Labels>& validation_labels, float64_t epsilon)
{
	auto current=get_root();
	prune_tree_from_current_node(validation_data->as<DenseFeatures<float64_t>>(),
			validation_labels->as<MulticlassLabels>(),current,epsilon);


}

SGVector<float64_t> C45ClassifierTree::get_certainty_vector() const
{
	return m_certainty;
}

void C45ClassifierTree::set_weights(SGVector<float64_t> w)
{
	m_weights=w;
	m_weights_set=true;
}

SGVector<float64_t> C45ClassifierTree::get_weights() const
{
	return m_weights;
}

void C45ClassifierTree::clear_weights()
{
	m_weights=SGVector<float64_t>();
	m_weights_set=false;
}

void C45ClassifierTree::set_feature_types(SGVector<bool> ft)
{
	m_nominal=ft;
	m_types_set=true;
}

SGVector<bool> C45ClassifierTree::get_feature_types() const
{
	return m_nominal;
}

void C45ClassifierTree::clear_feature_types()
{
	m_nominal=SGVector<bool>();
	m_types_set=false;
}

bool C45ClassifierTree::train_machine(std::shared_ptr<Features> data)
{
	require(data,"Data required for training");
	require(data->get_feature_class()==C_DENSE,"Dense data required for training");

	int32_t num_features=data->as<DenseFeatures<float64_t>>()->get_num_features();
	int32_t num_vectors=data->as<DenseFeatures<float64_t>>()->get_num_vectors();

	if (m_weights_set)
	{
		require(m_weights.vlen==num_vectors,"Length of weights vector (currently {}) should be same as"
					" number of vectors in data (presently {})",m_weights.vlen,num_vectors);
	}
	else
	{
		// all weights are equal to 1
		m_weights=SGVector<float64_t>(num_vectors);
		m_weights.fill_vector(m_weights.vector,m_weights.vlen,1.0);
	}

	if (m_types_set)
	{
		require(m_nominal.vlen==num_features,"Length of m_nominal vector (currently {}) should "
			"be same as number of features in data (presently {})",m_nominal.vlen,num_features);
	}
	else
	{
		io::warn("Feature types are not specified. All features are considered as continuous in training");
		m_nominal=SGVector<bool>(num_features);
		m_nominal.fill_vector(m_nominal.vector,m_nominal.vlen,false);
	}

	SGVector<int32_t> feature_ids(num_features);
	feature_ids.range_fill();

	set_root(C45train(data, m_weights, multiclass_labels(m_labels), feature_ids, 0));

	return true;
}

std::shared_ptr<TreeMachineNode<C45TreeNodeData>> C45ClassifierTree::C45train(const std::shared_ptr<Features>& data, SGVector<float64_t> weights,
	const std::shared_ptr<MulticlassLabels>& class_labels, SGVector<int32_t> feature_id_vector, int32_t level)
{
	require(data,"data matrix cannot be NULL");
	require(class_labels,"class labels cannot be NULL");
	auto node=std::make_shared<node_t>();
	auto feats=data->as<DenseFeatures<float64_t>>();
	int32_t num_vecs=feats->get_num_vectors();

	// set class_label for the node as the mode of occurring multiclass labels
	SGVector<float64_t> labels=class_labels->get_labels_copy();
	Math::qsort(labels);

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

	// if all remaining attributes are identical
	bool flag=true;
	for (int32_t i=1;i<num_vecs;i++)
	{
		for (int32_t j=0;j<feats->get_num_features();j++)
		{
			if (feats->get_feature_vector(i)[j]!=feats->get_feature_vector(i-1)[j])
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
			float64_t max_value=Math::MIN_REAL_NUMBER;
			for (int32_t k=0; k<num_vecs; k++)
			{
				feature_values[k]=(feats->get_feature_vector(k))[i];

				if (!Math::fequals(feature_values[k],MISSING,0) && feature_values[k]>max_value)
					max_value=feature_values[k];
			}

			for (int32_t k=0;k<num_vecs;k++)
			{
				if (feature_values[k]!=max_value && !Math::fequals(feature_values[k],MISSING,0))
				{
					// form temporary dense features to calculate gain (continuous->nominal conversion)
					float64_t z=feature_values[k];
					SGMatrix<float64_t> temp_feat_mat=SGMatrix<float64_t>(1,num_vecs);
					for (int32_t l=0;l<num_vecs;l++)
					{
						if (Math::fequals(feature_values[l],MISSING,0))
							temp_feat_mat(0,l)=MISSING;
						else if (feature_values[l]<=z)
							temp_feat_mat(0,l)=0.;
						else
							temp_feat_mat(0,l)=1.;
					}

					auto temp_feats=std::make_shared<DenseFeatures<float64_t>>(temp_feat_mat);
					float64_t gain=informational_gain_attribute(0,temp_feats,weights,class_labels);
					if (gain>max)
					{
						threshold=z;
						max=gain;
						best_feature_index=i;
					}
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
		for(int32_t p=0;p<num_vecs;p++)
		{
			feature_cache[p]=feats->get_feature_vector(p)[best_feature_index];
			if (Math::fequals(feature_cache[p],MISSING,0))
				continue;

			if (feature_cache[p]<=threshold)
				feats->get_feature_vector(p)[best_feature_index]=0.;
			else
				feats->get_feature_vector(p)[best_feature_index]=1.;
		}
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
		if (Math::fequals(best_feature_values[j],MISSING,0))
		{
			num_missing++;
			weight_missing+=weights[j];
		}
	}

	SGVector<float64_t> best_features_unique(num_vecs-num_missing);
	int32_t index=0;
	for (int32_t j=0;j<num_vecs;j++)
	{
		if (!Math::fequals(best_feature_values[j],MISSING,0))
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
			if (active_feature_value==best_feature_values[j] || Math::fequals(best_feature_values[j],MISSING,0))
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
			if (active_feature_value==sample[best_feature_index] || Math::fequals(sample[best_feature_index],MISSING,0))
			{
				int32_t idx=-1;
				for (int32_t k=0; k<sample.size(); k++)
				{
					if (k!=best_feature_index)
						mat(++idx, cnt) = sample[k];
				}

				new_labels_vector[cnt]=class_labels->get_labels()[j];
				if (!Math::fequals(sample[best_feature_index],MISSING,0))
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
			if (Math::fequals(best_feature_values[j],MISSING,0))
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
		auto new_class_labels=std::make_shared<MulticlassLabels>(new_labels_vector);
		auto new_data=std::make_shared<DenseFeatures<float64_t>>(mat);

		// recursion over child nodes
		auto child=C45train(new_data,new_weights,new_class_labels,new_feature_id_vector,level+1);
		node->data.attribute_id=feature_id_vector[best_feature_index];
		if (m_nominal[feature_id_vector[best_feature_index]])
			child->data.transit_if_feature_value=active_feature_value;
		else
			child->data.transit_if_feature_value=threshold;

		node->add_child(child);
	}

	// if continuous attribute - restoration required
	if (!m_nominal[feature_id_vector[best_feature_index]])
	{
		// restore data matrix
		for(int32_t p=0;p<num_vecs;p++)
			feats->get_feature_vector(p)[best_feature_index]=feature_cache[p];
	}

	return node;
}

void C45ClassifierTree::prune_tree_from_current_node(const std::shared_ptr<DenseFeatures<float64_t>>& feats,
		const std::shared_ptr<MulticlassLabels>& gnd_truth, const std::shared_ptr<node_t>& current, float64_t epsilon)
{
	// if leaf node then skip pruning
	if (current->data.attribute_id==-1)
		return;

	SGMatrix<float64_t> feature_matrix=feats->get_feature_matrix();
	auto children=current->get_children();

	if (m_nominal[current->data.attribute_id])
	{
		for (int32_t i=0; i<children.size(); i++)
		{
			// count number of feature vectors which transit into the child
			int32_t count=0;
			auto child=children[i];

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

			// prune the child subtree
			auto feats_prune = view(feats, subset);
			auto gt_prune = view(gnd_truth, subset);
			prune_tree_from_current_node(feats_prune, gt_prune, child, epsilon);
		}
	}
	else
	{
		require(children.size()==2,"The chosen attribute in current node is continuous. Expected number of"
					" children is 2 but current node has {} children.",children.size());

		auto left_child=children[0];
		auto right_child=children[1];

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
		if (count_left > 0)
		{
			auto feats_prune = view(feats, left_subset);
			auto gt_prune = view(gnd_truth, left_subset);
			// prune the left child subtree
			prune_tree_from_current_node(
			    feats_prune, gt_prune, left_child, epsilon);
		}

		// count_left is equal to num_cols if entire validation data in current node moves only to left child
		if (count_left<feature_matrix.num_cols)
		{
			auto feats_prune = view(feats, right_subset);
			auto gt_prune = view(gnd_truth, right_subset);

			// prune the right child subtree
			prune_tree_from_current_node(
			    feats_prune, gt_prune, right_child, epsilon);
		}
	}
	auto predicted_unpruned=apply_multiclass_from_current_node(feats, current);

	SGVector<float64_t> pruned_labels=SGVector<float64_t>(feature_matrix.num_cols);
	for (int32_t i=0; i<feature_matrix.num_cols; i++)
		pruned_labels[i]=current->data.class_label;

	auto predicted_pruned=std::make_shared<MulticlassLabels>(pruned_labels);


	auto accuracy=std::make_shared<MulticlassAccuracy>();
	float64_t unpruned_accuracy=accuracy->evaluate(predicted_unpruned, gnd_truth);
	float64_t pruned_accuracy=accuracy->evaluate(predicted_pruned, gnd_truth);

	if (unpruned_accuracy<pruned_accuracy+epsilon)
	{
		// set no children
		current->set_children({});

	}




}

float64_t C45ClassifierTree::informational_gain_attribute(int32_t attr_no, const std::shared_ptr<Features>& data,
				SGVector<float64_t> weights, const std::shared_ptr<MulticlassLabels>& class_labels)
{
	require(data,"Data required for information gain calculation");
	require(data->get_feature_class()==C_DENSE,
		"Dense data required for information gain calculation");

	float64_t gain=0;
	auto feats=data->as<DenseFeatures<float64_t>>();
	int32_t num_vecs=feats->get_num_vectors();
	SGVector<float64_t> gain_attribute_values;
	SGVector<float64_t> gain_weights=weights;
	auto gain_labels=class_labels;

	int32_t num_missing=0;
	for (int32_t i=0;i<num_vecs;i++)
	{
		if (Math::fequals((feats->get_feature_vector(i))[attr_no],MISSING,0))
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
			if (!Math::fequals((feats->get_feature_vector(i))[attr_no],MISSING,0))
			{
				gain_attribute_values[index]=(feats->get_feature_vector(i))[attr_no];
				gain_weights[index]=weights[i];
				label_vector[index++]=class_labels->get_label(i);
			}
		}

		num_vecs-=num_missing;
		gain_labels=std::make_shared<MulticlassLabels>(label_vector);
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

		auto sub_labels=std::make_shared<MulticlassLabels>(sub_class);
		float64_t sub_entropy=entropy(sub_labels,sub_weights);
		gain += sub_entropy*weight_count/total_weight;


	}

	float64_t data_entropy=entropy(gain_labels,gain_weights);
	gain = data_entropy-gain;

	if (num_missing!=0)
	{
		gain*=(num_vecs-0.f)/(num_vecs+num_missing-0.f);

	}

	return gain;
}

float64_t C45ClassifierTree::entropy(const std::shared_ptr<MulticlassLabels>& labels, SGVector<float64_t> weights)
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
		log_ratios[i] = std::log(log_ratios[i]);
	}

	return Statistics::entropy(log_ratios.vector,log_ratios.vlen);
}

std::shared_ptr<MulticlassLabels> C45ClassifierTree::apply_multiclass_from_current_node(const std::shared_ptr<DenseFeatures<float64_t>>& feats,
									const std::shared_ptr<node_t>& current, bool set_certainty)
{
	require(feats, "Features should not be NULL");
	require(current, "Current node should not be NULL");

	int32_t num_vecs=feats->get_num_vectors();
	SGVector<float64_t> labels(num_vecs);
	if (set_certainty)
		m_certainty=SGVector<float64_t>(num_vecs);

	// classify vectors in feature matrix taking one at a time
	for (int32_t i=0; i<num_vecs; i++)
	{
		// choose the current subtree as the entry point
		SGVector<float64_t> sample=feats->get_feature_vector(i);
		auto node=current;

		auto children=node->get_children();

		// traverse the subtree until leaf node is reached
		while (children.size())
		{
			bool flag=false;
			// if nominal attribute check for equality
			if (m_nominal[node->data.attribute_id])
			{
				for (int32_t j=0; j<children.size(); j++)
				{
					auto child=children[j];
					if (!child)
						error("{} element of children is NULL",j);

					if (child->data.transit_if_feature_value==sample[node->data.attribute_id])
					{
						flag=true;


						node=child;


						children=node->get_children();

						break;
					}


				}

				if (!flag)
					break;
			}
			// if not nominal attribute check if greater or less than threshold
			else
			{
				auto left_child=children[0];
				if (!left_child)
					error("left child is NULL");

				auto right_child=children[1];
				if (!right_child)
					error("right child is NULL");

				if (left_child->data.transit_if_feature_value>=sample[node->data.attribute_id])
				{

					node=left_child;



					children=node->get_children();
				}
				else
				{

					node=right_child;



					children=node->get_children();
				}



			}
		}

		// class_label of leaf node is the class to which chosen vector belongs
		labels[i]=node->data.class_label;

		if (set_certainty)
			m_certainty[i]=(node->data.total_weight-node->data.weight_minus)/node->data.total_weight;
	}

	return std::make_shared<MulticlassLabels>(labels);
}

void C45ClassifierTree::init()
{
	m_nominal=SGVector<bool>();
	m_weights=SGVector<float64_t>();
	m_certainty=SGVector<float64_t>();
	m_types_set=false;
	m_weights_set=false;

	SG_ADD(&m_nominal,"m_nominal", "feature types");
	SG_ADD(&m_weights,"m_weights", "weights");
	SG_ADD(&m_certainty,"m_certainty", "certainty");
	SG_ADD(&m_weights_set,"m_weights_set", "weights set");
	SG_ADD(&m_types_set,"m_types_set", "feature types set");
}

