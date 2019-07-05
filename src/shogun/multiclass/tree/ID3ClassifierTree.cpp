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
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/View.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/multiclass/tree/ID3ClassifierTree.h>

using namespace shogun;

ID3ClassifierTree::ID3ClassifierTree()
: TreeMachine<id3TreeNodeData>()
{
}

ID3ClassifierTree::~ID3ClassifierTree()
{
}

std::shared_ptr<MulticlassLabels> ID3ClassifierTree::apply_multiclass(std::shared_ptr<Features> data)
{
	require(data, "Data required for classification in apply_multiclass");

	auto current = get_root()->as<node_t>();
	return apply_multiclass_from_current_node(data->as<DenseFeatures<float64_t>>(), current);
}

bool ID3ClassifierTree::prune_tree(std::shared_ptr<DenseFeatures<float64_t>> validation_data,
			std::shared_ptr<MulticlassLabels> validation_labels, float64_t epsilon)
{
	auto current = get_root();
	prune_tree_machine(validation_data, validation_labels, current, epsilon);


	return true;
}

bool ID3ClassifierTree::train_machine(std::shared_ptr<Features> data)
{
	require(data,"Data required for training");
	require(data->get_feature_class()==C_DENSE, "Dense data required for training");

	int32_t num_features = data->as<DenseFeatures<float64_t>>()->get_num_features();
	SGVector<int32_t> feature_ids = SGVector<int32_t>(num_features);
	feature_ids.range_fill();

	set_root(id3train(data, multiclass_labels(m_labels), feature_ids, 0));

	return true;
}

std::shared_ptr<TreeMachineNode<id3TreeNodeData>> ID3ClassifierTree::id3train(std::shared_ptr<Features> data,
	std::shared_ptr<MulticlassLabels> class_labels, SGVector<int32_t> feature_id_vector, int32_t level)
{
	auto node = std::make_shared<node_t>();
	auto feats = data->as<DenseFeatures<float64_t>>();
	int32_t num_vecs = feats->get_num_vectors();

	// set class_label for the node as the mode of occurring multiclass labels
	SGVector<float64_t> labels = class_labels->get_labels_copy();
	Math::qsort(labels);

	int32_t most_label = labels[0];
	int32_t most_num = 1;
	int32_t count = 1;

	for (int32_t i=1; i<labels.vlen; i++)
	{
		if (labels[i] == labels[i-1])
		{
			count++;
		}
		else if (count>most_num)
		{
			most_num = count;
			most_label = labels[i-1];
			count = 1;
		}
		else
		{
			count = 1;
		}
	}

	node->data.class_label = most_label;

	// if all samples belong to the same class
	if (most_num == labels.vlen)
		return node;

	// if no feature is left
	if (feature_id_vector.vlen == 0)
		return node;

	// else get the feature with the highest informational gain
	float64_t max = 0;
	int32_t best_feature_index = -1;
	for (int32_t i=0; i<feats->get_num_features(); i++)
	{
		float64_t gain = informational_gain_attribute(i,feats,class_labels);

		if (gain >= max)
		{
			max = gain;
			best_feature_index = i;
		}
	}

	// get feature values for the best feature chosen
	SGVector<float64_t> best_feature_values = SGVector<float64_t>(num_vecs);
	for (int32_t i=0; i<num_vecs; i++)
		best_feature_values[i] = (feats->get_feature_vector(i))[best_feature_index];

	auto best_feature_labels = std::make_shared<MulticlassLabels>(best_feature_values);
	SGVector<float64_t> best_labels_unique = best_feature_labels->get_unique_labels();

	for (int32_t i=0; i<best_labels_unique.vlen; i++)
	{
		// compute the number of vectors with active attribute value
		int32_t num_cols = 0;
		float64_t active_feature_value = best_labels_unique[i];

		for (int32_t j=0; j<num_vecs; j++)
		{
			if ( active_feature_value == best_feature_values[j])
				num_cols++;
		}

		SGMatrix<float64_t> mat = SGMatrix<float64_t>(feats->get_num_features()-1, num_cols);
		SGVector<float64_t> new_labels_vector = SGVector<float64_t>(num_cols);

		int32_t cnt = 0;
		// choose the samples that have the active feature value
		for (int32_t j=0; j<num_vecs; j++)
		{
			SGVector<float64_t> sample = feats->get_feature_vector(j);
			if (active_feature_value == sample[best_feature_index])
			{
				int32_t idx = -1;
				for (int32_t k=0; k<sample.size(); k++)
				{
					if (k != best_feature_index)
						mat(++idx, cnt) = sample[k];
				}

				new_labels_vector[cnt] = class_labels->get_labels()[j];
				cnt++;
			}
		}

		// remove the best_attribute from the remaining attributes index vector
		SGVector<int32_t> new_feature_id_vector = SGVector<int32_t>(feature_id_vector.vlen-1);
		cnt = -1;
		for (int32_t j=0;j<feature_id_vector.vlen;j++)
		{
			if (j!=best_feature_index)
				new_feature_id_vector[++cnt] = feature_id_vector[j];
		}

		auto new_class_labels = std::make_shared<MulticlassLabels>(new_labels_vector);
		auto new_data = std::make_shared<DenseFeatures<float64_t>>(mat);

		auto child = id3train(new_data, new_class_labels, new_feature_id_vector, level+1);
		child->data.transit_if_feature_value = active_feature_value;
		node->data.attribute_id = feature_id_vector[best_feature_index];
		node->add_child(child);



	}



	return node;
}

float64_t ID3ClassifierTree::informational_gain_attribute(int32_t attr_no, std::shared_ptr<Features> data,
								std::shared_ptr<MulticlassLabels> class_labels)
{
	require(data,"Data required for information gain calculation");
	require(data->get_feature_class()==C_DENSE,
		"Dense data required for information gain calculation");

	float64_t gain = 0;
	auto feats = data->as<DenseFeatures<float64_t>>();
	int32_t num_vecs = feats->get_num_vectors();

	// get attribute values for attribute
	SGVector<float64_t> attribute_values = SGVector<float64_t>(num_vecs);

	for (int32_t i=0; i<num_vecs; i++)
		attribute_values[i] = (feats->get_feature_vector(i))[attr_no];

	auto attribute_labels = std::make_shared<MulticlassLabels>(attribute_values);
	SGVector<float64_t> attr_val_unique = attribute_labels->get_unique_labels();

	for (int32_t i=0; i<attr_val_unique.vlen; i++)
	{
		// calculate class entropy for the specific attribute_value
		int32_t attr_count=0;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (attribute_values[j] == attr_val_unique[i])
				attr_count++;
		}

		SGVector<float64_t> sub_class = SGVector<float64_t>(attr_count);
		int32_t count = 0;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (attribute_values[j] == attr_val_unique[i])
				sub_class[count++] = class_labels->get_label(j);
		}

		auto sub_labels = std::make_shared<MulticlassLabels>(sub_class);
		float64_t sub_entropy = entropy(sub_labels);
		gain += sub_entropy*(attr_count-0.f)/(num_vecs-0.f);


	}

	float64_t data_entropy = entropy(class_labels);
	gain = data_entropy-gain;



	return gain;
}

float64_t ID3ClassifierTree::entropy(std::shared_ptr<MulticlassLabels> labels)
{
	SGVector<float64_t> log_ratios = SGVector<float64_t>
			(labels->get_unique_labels().size());

	for (int32_t i=0;i<labels->get_unique_labels().size();i++)
	{
		int32_t count = 0;

		for (int32_t j=0;j<labels->get_num_labels();j++)
		{
			if (labels->get_unique_labels()[i] == labels->get_label(j))
					count++;
		}

		log_ratios[i] = (count-0.f)/(labels->get_num_labels()-0.f);

		if (log_ratios[i] != 0)
			log_ratios[i] = std::log(log_ratios[i]);
	}

	return Statistics::entropy(log_ratios.vector, log_ratios.vlen);
}

void ID3ClassifierTree::prune_tree_machine(std::shared_ptr<DenseFeatures<float64_t>> feats,
		std::shared_ptr<MulticlassLabels> gnd_truth, std::shared_ptr<node_t> current, float64_t epsilon)
{
	SGMatrix<float64_t> feature_matrix = feats->get_feature_matrix();
	auto children = current->get_children();

	for (int32_t i=0; i<children.size(); i++)
	{
		// count number of feature vectors which transit into the child
		int32_t count = 0;
		auto child = children[i];

		for (int32_t j=0; j<feature_matrix.num_cols; j++)
		{
			float child_transit = child->data.transit_if_feature_value;

			if (child_transit == feature_matrix(current->data.attribute_id,j))
				count++;
		}

		// form new subset of features and labels
		SGVector<index_t> subset = SGVector<index_t>(count);
		int32_t k = 0;

		for (int32_t j=0; j<feature_matrix.num_cols;j++)
		{
			float child_transit = child->data.transit_if_feature_value;

			if (child_transit == feature_matrix(current->data.attribute_id,j))
			{
				subset[k] = (index_t) j;
				k++;
			}
		}

		auto feats_train = view(feats, subset);
		auto gt_train = view(gnd_truth, subset);

		// prune the child subtree
		prune_tree_machine(feats_train, gt_train, child, epsilon);
	}
	auto predicted_unpruned = apply_multiclass_from_current_node(feats, current);

	SGVector<float64_t> pruned_labels = SGVector<float64_t>(feature_matrix.num_cols);
	for (int32_t i=0; i<feature_matrix.num_cols; i++)
		pruned_labels[i] = current->data.class_label;

	auto predicted_pruned = std::make_shared<MulticlassLabels>(pruned_labels);


	auto accuracy = std::make_shared<MulticlassAccuracy>();
	float64_t unpruned_accuracy = accuracy->evaluate(predicted_unpruned, gnd_truth);
	float64_t pruned_accuracy = accuracy->evaluate(predicted_pruned, gnd_truth);

	if (unpruned_accuracy<pruned_accuracy+epsilon)
	{
		// set no children
		current->set_children({});
	}
}

std::shared_ptr<MulticlassLabels> ID3ClassifierTree::apply_multiclass_from_current_node(std::shared_ptr<DenseFeatures<float64_t>> feats,
											std::shared_ptr<node_t> current)
{
	require(feats, "Features should not be NULL");
	require(current, "Current node should not be NULL");

	int32_t num_vecs = feats->get_num_vectors();
	SGVector<float64_t> labels = SGVector<float64_t>(num_vecs);

	// classify vectors in feature matrix taking one at a time
	for (int32_t i=0; i<num_vecs; i++)
	{
		// choose the current subtree as the entry point
		SGVector<float64_t> sample = feats->get_feature_vector(i);
		auto node = current;

		auto children = node->get_children();

		// traverse the subtree until leaf node is reached
		while (children.size())
		{
			bool flag = false;
			for (int32_t j=0; j<children.size(); j++)
			{
				auto child = children[j];
				if (child->data.transit_if_feature_value
						== sample[node->data.attribute_id])
				{
					flag = true;


					node = child;


					children = node->get_children();

					break;
				}
			}

			if (!flag)
				break;
		}

		// class_label of leaf node is the class to which chosen vector belongs
		labels[i] = node->data.class_label;
	}

	return std::make_shared<MulticlassLabels>(labels);
}
