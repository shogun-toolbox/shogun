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
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/util/zip_iterator.h>

using namespace shogun;

Perceptron::Perceptron() : IterativeMachine<LinearMachine>()
{
	m_max_iterations = 1000;
	learn_rate = 0.1;
	m_initialize_hyperplane = true;
	SG_ADD(
	    &m_initialize_hyperplane, "initialize_hyperplane",
	    "Whether to initialize hyperplane.", ParameterProperties::HYPER)
	SG_ADD(&learn_rate, "learn_rate", "Learning rate.", ParameterProperties::HYPER)
}

Perceptron::~Perceptron()
{
}

void Perceptron::init_model(const std::shared_ptr<DotFeatures>& features)
{
	int32_t num_feat = features->get_dim_feature_space();

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

void Perceptron::iteration(
    const std::shared_ptr<DotFeatures>& features, const std::shared_ptr<Labels>& labs)
{
	bool converged = true;
	SGVector<float64_t> w = get_w();

	auto labels = labs->as<BinaryLabels>()->get_int_labels();
	for (const auto& [v, true_label] : zip_iterator(DotIterator(features), labels))
	{
		const auto predicted_label = v.dot(w) + bias;

		if (Math::sign<float64_t>(predicted_label) != true_label)
		{
			converged = false;
			const auto gradient = learn_rate * true_label;
			bias += gradient;
			v.add(gradient, w);

			observe<SGVector<float64_t>>(m_current_iteration, "w");
			observe<float64_t>(m_current_iteration, "bias");
		}
	}
	m_complete = converged;
}

void Perceptron::set_initialize_hyperplane(bool initialize_hyperplane)
{
	m_initialize_hyperplane = initialize_hyperplane;
}

bool Perceptron::get_initialize_hyperplane()
{
	return m_initialize_hyperplane;
}
