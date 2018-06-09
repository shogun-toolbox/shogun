/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Giovanni De Toni, 
 *          Michele Mazzoni, Heiko Strathmann, Fernando Iglesias
 */

#include <shogun/base/progress.h>
#include <shogun/base/range.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CPerceptron::CPerceptron() : CIterativeLinearMachine()
{
	init();
}

void CPerceptron::init()
{
	m_max_iterations = 1000;
	learn_rate = 0.1;
	m_initialize_hyperplane = true;
	SG_ADD(
	    &m_initialize_hyperplane, "initialize_hyperplane",
	    "Whether to initialize hyperplane.", MS_AVAILABLE);
	SG_ADD(&learn_rate, "learn_rate", "Learning rate.", MS_AVAILABLE);
}

CPerceptron::~CPerceptron()
{
}

void CPerceptron::init_model(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		m_continue_features = data;
	}

	ASSERT(m_continue_features)

	int32_t num_feat =
	    m_continue_features->as<CDotFeatures>()->get_dim_feature_space();

	SGVector<float64_t> w;
	if (m_initialize_hyperplane)
	{
		w = SGVector<float64_t>(num_feat);
		set_w(w);

		//start with uniform w, bias=0
		w.set_const(1.0 / num_feat);
		bias=0;
	}
}

void CPerceptron::iteration()
{
	bool converged = true;
	SGVector<float64_t> w = get_w();
	SGVector<float64_t> output(m_continue_features->get_num_vectors());

	auto labels = binary_labels(m_labels)->get_int_labels();
	auto iter_train_labels = labels.begin();
	auto iter_output = output.begin();

	for (const auto& v : DotIterator(m_continue_features->as<CDotFeatures>()))
	{
		const auto true_label = *(iter_train_labels++);
		auto& predicted_label = *(iter_output++);

		predicted_label = v.dot(w) + bias;

		if (CMath::sign<float64_t>(predicted_label) != true_label)
		{
			converged = false;
			const auto gradient = learn_rate * true_label;
			bias += gradient;
			v.add(gradient, w);
		}
	}
	m_complete = converged;
}

void CPerceptron::set_initialize_hyperplane(bool initialize_hyperplane)
{
	m_initialize_hyperplane = initialize_hyperplane;
}

bool CPerceptron::get_initialize_hyperplane()
{
	return m_initialize_hyperplane;
}
