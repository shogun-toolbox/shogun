/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Giovanni De Toni,
 *          Michele Mazzoni
 */

#include <shogun/classifier/AveragedPerceptron.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CAveragedPerceptron::CAveragedPerceptron() : CIterativeMachine<CLinearMachine>()
{
	init();
}

CAveragedPerceptron::~CAveragedPerceptron()
{
}

void CAveragedPerceptron::init()
{
	set_max_iter(1000);
	set_learn_rate(0.1);

	SG_ADD(
	    &learn_rate, "learn_rate", "Learning rate.",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &cached_w, "cached_w",
	    "Cached weights that contribute to the average.");
	SG_ADD(
	    &cached_bias, "cached_bias",
	    "Cached bias that contribute to the average.");
}

void CAveragedPerceptron::init_model(CFeatures* data)
{
	ASSERT(m_labels)
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
		set_continue_features(data);
	}
	ASSERT(features)

	SGVector<int32_t> train_labels = binary_labels(m_labels)->get_int_labels();
	int32_t num_feat = features->get_dim_feature_space();
	int32_t num_vec = features->get_num_vectors();
	ASSERT(num_vec == train_labels.vlen)

	SGVector<float64_t> w(num_feat);
	cached_w = SGVector<float64_t>(num_feat);
	// start with uniform w, bias=0, tmp_bias=0
	bias = 0;
	cached_bias = 0;
	cached_w.set_const(1.0 / num_feat);
	w.set_const(0);
	set_w(w);
}

void CAveragedPerceptron::iteration()
{
	bool converged = true;
	
	SGVector<float64_t> w = get_w();
	auto labels = binary_labels(m_labels)->get_int_labels();
	auto iter_train_labels = labels.begin();
	
	int32_t num_vec = features->get_num_vectors();
	// this assumes that m_current_iteration starts at 0
	int32_t num_prev_weights = num_vec * m_current_iteration + 1;
	
	for (const auto& feature : DotIterator(features))
	{
		const auto true_label = *(iter_train_labels++);
		auto predicted_label = feature.dot(cached_w) + cached_bias;

		if (CMath::sign<float64_t>(predicted_label) != true_label)
		{
			converged = false;
			const auto gradient = learn_rate * true_label;
			cached_bias += gradient;
			feature.add(gradient, cached_w);
		}
		linalg::update_mean(w, cached_w, num_prev_weights);
		linalg::update_mean(bias, cached_bias, num_prev_weights);
		num_prev_weights++;
	}
	m_complete = converged;
}