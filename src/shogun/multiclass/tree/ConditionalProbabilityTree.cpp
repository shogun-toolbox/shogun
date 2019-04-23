/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Sanuj Sharma, Sergey Lisitsyn,
 *          Viktor Gal
 */

#include <vector>
#include <stack>

#include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
#include <shogun/classifier/svm/OnlineLibLinear.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;
using namespace std;

std::shared_ptr<MulticlassLabels> ConditionalProbabilityTree::apply_multiclass(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STREAMING_DENSE)
			SG_ERROR("Expected StreamingDenseFeatures\n")
		if (data->get_feature_type() != F_SHORTREAL)
			SG_ERROR("Expected float32_t feature type\n")

		set_features(data->as<StreamingDenseFeatures<float32_t>>());
	}

	vector<int32_t> predicts;

	m_feats->start_parser();
	while (m_feats->get_next_example())
	{
		predicts.push_back(apply_multiclass_example(m_feats->get_vector()));
		m_feats->release_example();
	}
	m_feats->end_parser();

	auto labels = std::make_shared<MulticlassLabels>(predicts.size());
	for (size_t i=0; i < predicts.size(); ++i)
		labels->set_int_label(i, predicts[i]);
	return labels;
}

int32_t ConditionalProbabilityTree::apply_multiclass_example(SGVector<float32_t> ex)
{
	compute_conditional_probabilities(ex);
	SGVector<float64_t> probs(m_leaves.size());
	for (auto it = m_leaves.begin(); it != m_leaves.end(); ++it)
	{
		probs[it->first] = accumulate_conditional_probability(it->second);
	}
	return Math::arg_max(probs.vector, 1, probs.vlen);
}

void ConditionalProbabilityTree::compute_conditional_probabilities(SGVector<float32_t> ex)
{
	stack<std::shared_ptr<bnode_t>> nodes;
	nodes.push(m_root->as<bnode_t>());

	while (!nodes.empty())
	{
		auto node = nodes.top();
		nodes.pop();
		if (node->left())
		{
			nodes.push(node->left());
			nodes.push(node->right());

			// don't calculate for leaf
			node->data.p_right = predict_node(ex, node);
		}
	}
}

float64_t ConditionalProbabilityTree::accumulate_conditional_probability(std::shared_ptr<bnode_t> leaf)
{
	float64_t prob = 1;
	auto par = leaf->parent()->as<bnode_t>();
	while (par != NULL)
	{
		if (leaf == par->left())
			prob *= (1-par->data.p_right);
		else
			prob *= par->data.p_right;

		leaf = par;
		par = leaf->parent()->as<bnode_t>();
	}

	return prob;
}

bool ConditionalProbabilityTree::train_machine(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STREAMING_DENSE)
			SG_ERROR("Expected StreamingDenseFeatures\n")
		if (data->get_feature_type() != F_SHORTREAL)
			SG_ERROR("Expected float32_t features\n")
		set_features(data->as<StreamingDenseFeatures<float32_t>>());
	}
	else
	{
		if (!m_feats)
			SG_ERROR("No data features provided\n")
	}

	m_machines.clear();

	m_root = NULL;

	m_leaves.clear();

	m_feats->start_parser();
	for (int32_t ipass=0; ipass < m_num_passes; ++ipass)
	{
		while (m_feats->get_next_example())
		{
			train_example(m_feats, static_cast<int32_t>(m_feats->get_label()));
			m_feats->release_example();
		}

		if (ipass < m_num_passes-1)
			m_feats->reset_stream();
	}
	m_feats->end_parser();

	for (auto m: m_machines)
	{
		auto lll = m->as<OnlineLibLinear>();
		lll->stop_train();
	}

	return true;
}

void ConditionalProbabilityTree::train_example(std::shared_ptr<StreamingDenseFeatures<float32_t>> ex, int32_t label)
{
	if (!m_root)
	{
		auto root = std::make_shared<bnode_t>();
		m_root = root;
		m_root->data.label = label;
		m_leaves.emplace(label, root);
		m_root->machine(create_machine(ex));
		return;
	}

	if (m_leaves.find(label) != m_leaves.end())
	{
		train_path(ex, m_leaves[label]);
	}
	else
	{
		auto node = m_root->as<bnode_t>();
		while (node->left() != NULL)
		{
			// not a leaf
			bool is_left = which_subtree(node, ex->get_vector());
			float64_t node_label;
			if (is_left)
				node_label = 0;
			else
				node_label = 1;
			train_node(ex, node_label, node);

			if (is_left)
				node = node->left();
			else
				node = node->right();
		}

		m_leaves.erase(node->data.label);

		auto left_node = std::make_shared<bnode_t>();
		left_node->data.label = node->data.label;
		node->data.label = -1;
		auto node_mch = m_machines.at(node->machine())->as<OnlineLibLinear>();
		auto mch = std::make_shared<OnlineLibLinear>(node_mch);

		mch->start_train();
		m_machines.push_back(mch);
		left_node->machine(m_machines.size()-1);
		m_leaves.emplace(left_node->data.label, left_node);
		node->left(left_node);

		auto right_node = std::make_shared<bnode_t>();
		right_node->data.label = label;
		right_node->machine(create_machine(ex));
		m_leaves.emplace(label, right_node);
		node->right(right_node);
	}
}

void ConditionalProbabilityTree::train_path(std::shared_ptr<StreamingDenseFeatures<float32_t>> ex, std::shared_ptr<bnode_t> node)
{
	float64_t node_label = 0;
	train_node(ex, node_label, node);

	auto par = node->parent()->as<bnode_t>();
	while (par != NULL)
	{
		if (par->left() == node)
			node_label = 0;
		else
			node_label = 1;

		train_node(ex, node_label, par);
		node = par;
		par = node->parent()->as<bnode_t>();
	}
}

void ConditionalProbabilityTree::train_node(std::shared_ptr<StreamingDenseFeatures<float32_t>> ex, float64_t label, std::shared_ptr<bnode_t> node)
{
	REQUIRE(node, "Node must not be NULL\n");
	auto mch = m_machines.at(node->machine())->as<OnlineLibLinear>();
	REQUIRE(mch, "Instance of %s could not be casted to OnlineLibLinear\n", node->get_name());
	mch->train_example(ex, label);

}

float64_t ConditionalProbabilityTree::predict_node(SGVector<float32_t> ex, std::shared_ptr<bnode_t> node)
{
	REQUIRE(node, "Node must not be NULL\n");
	auto mch = m_machines.at(node->machine())->as<OnlineLibLinear>();
	REQUIRE(mch, "Instance of %s could not be casted to OnlineLibLinear\n", node->get_name());
	float64_t pred = mch->apply_one(ex.vector, ex.vlen);

	// use sigmoid function to turn the decision value into valid probability
	return 1.0 / (1 + std::exp(-pred));
}

int32_t ConditionalProbabilityTree::create_machine(std::shared_ptr<StreamingDenseFeatures<float32_t>> ex)
{
	auto mch = std::make_shared<OnlineLibLinear>();
	mch->start_train();
	mch->train_example(ex, 0);
	m_machines.push_back(mch);
	return m_machines.size()-1;
}
